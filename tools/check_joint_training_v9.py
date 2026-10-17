import json,time,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_joint_v9')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def live(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
saved=root/'current_training_status.json';previous=read(saved)
if previous and previous.get('stable') and time.time()-previous['time']<600:
    print(json.dumps({'skipped':'stable training minimum600 seconds','last_time':previous['time']}));raise SystemExit
controller=read(root/'controller.json');out=root/'runs/joint';supervisor=read(out/'supervisor.json')
r={'time':time.time(),'controller':controller,'controller_live':live((controller or {}).get('pid')),
   'allocation':read(root/'allocation.json'),'supervisor':supervisor,'live':live((supervisor or {}).get('pid')),
   'status':read(out/'status.json'),'checkpoint':read(out/'latest/COMMITTED'),'free_gib':shutil.disk_usage(root).free/2**30,'ranks':[]}
for p in sorted(out.glob('rank*.jsonl')):
    lines=p.read_text().splitlines()
    if lines:
        row=json.loads(lines[-1]);r['ranks'].append({k:row.get(k) for k in ['rank','update','episodes','successes','seconds','losses','symbolic_audit','gradient_by_module','gpu_peak_gib']})
r['evaluations']={f.name:{s:read(f/s/'summary.json') for s in ['development','fixed80']} for f in sorted((out/'evaluations').glob('step_*'))[-2:]}
r['controller_tail']=(root/'controller.log').read_text(errors='replace')[-1200:] if (root/'controller.log').exists() else None
logs=sorted(out.glob('launch*.log'),key=lambda p:p.stat().st_mtime)
r['log_tail']=logs[-1].read_text(errors='replace')[-1800:] if logs else None
r['stable']=bool(r['live'] and (r['status'] or {}).get('update',0)>0)
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
saved.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
