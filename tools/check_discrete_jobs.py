import json,time,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/discrete_v4')
saved=root/'runs/current_status.json'
if saved.exists():
    previous=json.loads(saved.read_text());active=[v for k,v in previous.get('jobs',{}).items() if not k.endswith('_smoke')]
    if active and all(v.get('live_command') and v.get('status.json',{}).get('update',0)>=1 for v in active) and time.time()-previous['time']<600:
        print(json.dumps({'skipped':'stable training minimum polling interval is 600s','last_time':previous['time']}))
        raise SystemExit(0)
result={'time':time.time(),'jobs':{}}
for out in sorted((root/'runs').glob('*')):
    if not out.is_dir():continue
    record={}
    for name in ['process.json','supervisor.json','status.json','latest/COMMITTED','architecture.json']:
        p=out/name
        if p.exists():record[name]=json.loads(p.read_text())
    pid=record.get('supervisor.json',record.get('process.json',{})).get('pid')
    proc=Path(f'/proc/{pid}/cmdline')
    record['live_command']=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
    logs=sorted(out.glob('launch*.log'),key=lambda p:p.stat().st_mtime)
    if logs:record['log_tail']=logs[-1].read_text(errors='replace')[-3800:]
    f=out/'rank0.jsonl'
    if f.exists():record['last_rank_record']=json.loads(f.read_text().splitlines()[-1])
    record['evaluations']={}
    for step in sorted((out/'evaluations').glob('step_*'))[-2:]:
        record['evaluations'][step.name]={suite:json.loads((step/suite/'summary.json').read_text())
            for suite in ['development','fixed80'] if (step/suite/'summary.json').exists()}
    result['jobs'][out.name]=record
result['gpu']=subprocess.check_output(['nvidia-smi','--id=6,7','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
(root/'runs/current_status.json').write_text(json.dumps(result,indent=2))
compact={}
for name,v in result['jobs'].items():
    r=v.get('last_rank_record',{})
    compact[name]={'live_command':v['live_command'],'status':v.get('status.json'),
        'supervisor':v.get('supervisor.json'),'checkpoint':v.get('latest/COMMITTED'),
        'seconds':r.get('seconds'),'phases':r.get('phase_seconds'),'audit':r.get('symbolic_audit'),
        'gpu_peak_gib':r.get('gpu_peak_gib'),'gradient_by_module':r.get('gradient_by_module'),
        'losses':r.get('losses'),'evaluations':v['evaluations']}
    if not r or 'Traceback' in v.get('log_tail',''):compact[name]['log_tail']=v.get('log_tail')
print(json.dumps({'time':result['time'],'jobs':compact,'gpu':result['gpu']},indent=2))
