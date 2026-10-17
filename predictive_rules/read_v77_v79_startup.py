import json,time,subprocess
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
result=dict(time=time.time(),runs={})
for name in ('memory_dynamics_v77','joint_memory_dynamics_v78','full_history_v79'):
    p=ROOT/('run_'+name);r={}
    for f in ('supervisor.json','status_rank0.json','status_rank1.json','capacity.json','result.json'):
        if (p/f).exists():
            row=json.loads((p/f).read_text());pid=row.get('pid');proc=Path(f'/proc/{pid}/cmdline')
            if pid:row['actual_cmdline']=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
            r[f]=row
    for f in ('training.log','rank0.jsonl','heldout.jsonl'):
        if (p/f).exists():r[f]=(p/f).read_text(errors='replace').splitlines()[-3:]
    result['runs'][name]=r
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
(ROOT/'v77_v79_startup.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
