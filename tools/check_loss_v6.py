import json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_loss_v6');saved=root/'current_status.json'
def read(path):return json.loads(path.read_text()) if path.exists() else None
def process(pid):
    path=Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else None
previous=read(saved)
if previous and previous.get('stable') and time.time()-previous['time']<600:
    print(json.dumps({'skipped':'stable training minimum 600 seconds','last_time':previous['time']}));raise SystemExit(0)
controller=read(root/'controller.json');allocation=read(root/'allocation.json')
result={'time':time.time(),'controller':controller,'controller_process':process(controller['pid']),
    'allocation':allocation,'jobs':{}}
for name in ['loss_reference','loss_balanced']:
    out=root/'runs'/name;record={key:read(out/f'{key}.json') for key in ['status','supervisor','initialization']}
    record['checkpoint']=read(out/'latest/COMMITTED')
    record['live_command']=process((record['supervisor'] or {}).get('pid'))
    rank=out/'rank0.jsonl'
    if rank.exists():
        last=json.loads(rank.read_text().splitlines()[-1])
        record['training']={k:last.get(k) for k in ['update','seconds','training_frames','demonstrations',
            'gpu_peak_gib','losses','gradient_by_module']}
    record['evaluations']={}
    for step in sorted((out/'evaluations').glob('step_*'))[-2:]:
        record['evaluations'][step.name]={suite:read(step/suite/'summary.json') for suite in ['development','fixed80']}
    logs=sorted(out.glob('launch*.log'),key=lambda p:p.stat().st_mtime)
    if logs:record['log_tail']=logs[-1].read_text(errors='replace')[-2200:]
    result['jobs'][name]=record
result['stable']=all(v['live_command'] and (v['status'] or {}).get('update',0)>0 for v in result['jobs'].values())
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
result['controller_tail']=(root/'controller.log').read_text(errors='replace')[-3000:]
saved.write_text(json.dumps(result,indent=2))
for job in result['jobs'].values():
    if job['live_command'] and (job['status'] or {}).get('update',0)>0:job.pop('log_tail',None)
print(json.dumps(result,indent=2))
