import json,time,subprocess
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=r/'run_raw_grounding_v31'
launcher=json.loads((r/'raw_grounding_v31_launcher.json').read_text());p=Path('/proc')/str(launcher['pid'])/'cmdline'
record=dict(time=time.time(),pid=launcher['pid'],process_present=p.exists(),command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None)
for name in ['status.json','latest.COMMITTED.json']:
    path=out/name;record[name]=json.loads(path.read_text()) if path.exists() else None
record['log_tail']=(r/'raw_grounding_v31.log').read_text(errors='replace')[-7000:]
workers=json.loads((r/'raw_successor_features_v31/parallel_launcher.json').read_text())['workers']
record['collectors']=[dict(pid=w['pid'],alive=Path('/proc',str(w['pid']),'cmdline').exists()) for w in workers]
record['raw_training_positions']=sum(json.loads(f.read_text())['positions'] for f in (r/'raw_successor_features_v31').glob('counterfactual_*/*.json'))
record['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
(r/'raw_grounding_v31_startup.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
