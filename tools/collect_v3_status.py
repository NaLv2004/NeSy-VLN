import json,time,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/v3');run=root/'runs/full_v3'
output=root/'runs/current_status.json'
def read(p):return json.loads(p.read_text()) if p.exists() else None
previous=read(output)
if previous and time.time()-previous['time']<600:
    print(json.dumps({'skipped':'minimum 600-second polling interval','last_time':previous['time']}))
    raise SystemExit(0)
supervisor=read(run/'supervisor.json');pid=supervisor.get('pid') if supervisor else None
proc=Path(f'/proc/{pid}/cmdline') if pid else None
command=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc and proc.exists() else None
ranks=[]
for r in range(6):
    f=run/f'rank{r}.jsonl'
    if f.exists():ranks.append(json.loads(f.read_text().splitlines()[-1]))
evals={}
for folder in sorted((run/'evaluations').glob('step_*'))[-3:]:
    evals[folder.name]={suite:read(folder/suite/'summary.json') for suite in ['development','fixed80']}
result={'time':time.time(),'supervisor':supervisor,'live_command':command,'status':read(run/'status.json'),
    'checkpoint':read(run/'latest/COMMITTED'),'ranks':ranks,'evaluations':evals,
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
    'disk':subprocess.check_output(['df','-h',str(root)],text=True)}
logs=sorted(run.glob('launch_*.log'),key=lambda p:p.stat().st_mtime)
if logs:result['log_tail']=logs[-1].read_text(errors='replace')[-2500:]
output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='ranks'},indent=2))
