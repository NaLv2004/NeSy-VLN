import json,time,subprocess
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=r/'run_rule_planning_v32'
launch=json.loads((r/'rule_planning_v32_launcher.json').read_text());pid=launch['pid'];p=Path('/proc',str(pid),'cmdline')
def read(path):return json.loads(path.read_text()) if path.exists() else None
record=dict(observed_at=time.time(),pid=pid,process_alive=p.exists(),command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
    status=read(out/'status.json'),committed=read(out/'latest.COMMITTED.json'),
    source_checkpoint=launch['source_checkpoint'],log_tail=(r/'rule_planning_v32.log').read_text(errors='replace')[-6000:],
    collector_status=read(r/'raw_successor_features_v31/parallel_render/status.json'),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
(r/'rule_planning_v32_startup.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
