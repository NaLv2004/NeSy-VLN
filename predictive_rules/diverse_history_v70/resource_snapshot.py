import json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
def read(p):return json.loads(p.read_text()) if p.exists() else {}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
pids={int(line.split(',')[1].strip()) for line in apps.splitlines() if line.strip()}
collection=read(root/'diverse_history_v70/status.json');training=read(root/'run_expanded_dynamics_v71/supervisor.json')
pids.update(p for p in [collection.get('pid'),training.get('pid'),training.get('child'),*collection.get('workers',[])] if p)
processes=[dict(pid=p,command=Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{p}/cmdline').exists() else '') for p in sorted(pids)]
record=dict(time=time.time(),gpus=gpu,apps=apps,processes=processes,collection=collection,training=training,
    ranks=[read(root/f'diverse_history_v70/status_rank{r}.json') for r in (0,1)])
(root/'diverse_history_v70/resource_snapshot.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
