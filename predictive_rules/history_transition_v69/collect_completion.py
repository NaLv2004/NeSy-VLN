"""Wait for the exact supervisor's exit; no repetitive status polling."""
import json,os,select,time,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_history_transition_v69')
supervisor=json.loads((root/'supervisor.json').read_text());pid=supervisor['pid']
try:
    handle=os.pidfd_open(pid)
except ProcessLookupError:handle=None
if handle is not None:
    select.select([handle],[],[],45);os.close(handle)
supervisor=json.loads((root/'supervisor.json').read_text())
result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None
rows=[json.loads(v) for v in (root/'heldout.jsonl').read_text().splitlines()]
best=min(rows,key=lambda x:x['nll'])
pids=[pid,*[json.loads((root/f'status_rank{r}.json').read_text())['pid'] for r in (0,1)]]
processes=[dict(pid=p,command=Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{p}/cmdline').exists() else '') for p in pids]
record=dict(time=time.time(),supervisor=supervisor,result=result,processes=processes,baseline=rows[0],
    best_heldout=best,last=rows[-1],gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
(root/'completion_observation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
