import json,time,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/v2');run=root/'runs/full_v2'
def read(p):return json.loads(p.read_text()) if p.exists() else None
supervisor=read(run/'supervisor.json');pid=supervisor.get('pid') if supervisor else None
proc=Path(f'/proc/{pid}/cmdline') if pid else None
command=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc and proc.exists() else None
ranks=[]
for r in range(6):
    f=run/f'rank{r}.jsonl'
    if f.exists():
        row=json.loads(f.read_text().splitlines()[-1])
        ranks.append({k:row.get(k) for k in ['rank','update','seconds','successes','episodes','symbolic_audit','losses',
            'gradient_by_module','gpu_peak_gib','gpu_reserved_peak_gib','training_chunk_batch','next_training_chunk_batch','phase_seconds']})
evals={}
for folder in sorted((run/'evaluations').glob('step_*'))[-3:]:
    evals[folder.name]={suite:read(folder/suite/'summary.json') for suite in ['development','fixed80']}
result={'time':time.time(),'supervisor':supervisor,'live_command':command,'status':read(run/'status.json'),
        'checkpoint':read(run/'latest/COMMITTED'),'ranks':ranks,'evaluations':evals,
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
        'disk':subprocess.check_output(['df','-h',str(root)],text=True)}
logs=sorted(run.glob('launch_*.log'),key=lambda p:p.stat().st_mtime)
if logs:result['log_tail']=logs[-1].read_text(errors='replace')[-2000:]
(root/'runs/current_status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
