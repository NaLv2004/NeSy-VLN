import json,time,subprocess
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
result=dict(time=time.time(),runs={})
for name in ('stable_rules_v75','expanded_history_v76'):
    out=ROOT/('run_'+name);r={}
    for fn in ('supervisor.json','status_rank0.json','status_rank1.json','dataset.json','capacity.json','baseline_holdout.json','result.json'):
        p=out/fn
        if p.exists():
            row=json.loads(p.read_text())
            if fn=='dataset.json':row={k:v for k,v in row.items() if k not in ('heldout_episode_ids',)}
            r[fn]=row
    for fn in ('training.log','rank0.jsonl','heldout.jsonl'):
        p=out/fn
        if p.exists():r[fn]=p.read_text(errors='replace').splitlines()[-1:]
    for row in list(r.values()):
        if isinstance(row,dict) and row.get('pid'):
            p=Path('/proc')/str(row['pid'])/'cmdline';row['actual_cmdline']=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
    result['runs'][name]=r
result['gpu']=subprocess.check_output(['nvidia-smi','-i','2,3,4,5','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
(ROOT/'parallel_v75_v76_observation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
