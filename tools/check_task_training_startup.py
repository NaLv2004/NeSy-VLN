import json,shutil,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_task_v15');out=root/'runs/paired_tasks'
def read(p):return json.loads(p.read_text()) if p.exists() else None
supervisor=read(out/'supervisor.json');pid=(supervisor or {}).get('pid')
p=Path(f'/proc/{pid}/cmdline')
lines=(out/'rank0.jsonl').read_text().splitlines() if (out/'rank0.jsonl').exists() else []
result={'time':time.time(),'supervisor':supervisor,'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
    'status':read(out/'status.json'),'commit':read(out/'latest/COMMITTED'),'initialization':read(out/'initialization.json'),
    'last_update':json.loads(lines[-1]) if lines else None,
    'tail':{p.name:p.read_text(errors='replace')[-2200:] for p in sorted(out.glob('launch_*.log'))[-2:]},
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
    'free_gib':shutil.disk_usage(root).free/2**30}
(root/'startup_inspection.json').write_text(json.dumps(result,indent=2))
result.pop('last_update',None);print(json.dumps(result,indent=2))
