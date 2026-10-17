import json,time,shutil
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');read=lambda p:json.loads(p.read_text()) if p.exists() else None
rows=[]
for run in [b/'symbolic_execution_v38/runs/detached_outcome_control',b/'symbolic_execution_v38/runs/grounded_outcome_feedback',b/'token_revision_rl_v45']:
 s=read(run/'supervisor.json');pid=(s or {}).get('pid');proc=Path(f'/proc/{pid}/cmdline')
 rows.append({'run':str(run),'supervisor':s,'status':read(run/'status.json'),'commit':read(run/'latest/COMMITTED'),'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,'storage_waits':{str(p.relative_to(run)):read(p) for p in run.rglob('WAITING_FOR_STORAGE.json')}})
r={'time':time.time(),'reason':'Confirm verified storage stall cleared after cold checkpoint compaction','free_gib':shutil.disk_usage(b).free/2**30,'jobs':rows};(b/'storage_repair_verification_v46.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
p=next((b/'spatial_automaton_v43/development').glob('*/trace.json'));r=read(p);step=r['steps'][0]
print(json.dumps({'trace_keys':list(r),'step_keys':list(step),'step_without_proof':{k:v for k,v in step.items() if k not in ['symbols','proof','probabilities']},'metrics':r['metrics'],'evaluation':read(p.parent/'evaluation.json')},indent=2))
