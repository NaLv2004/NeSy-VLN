import json,subprocess,time
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
result={'time':time.time(),'processes':{}}
for pid in [1750864,1833024]:
    p=Path('/proc')/str(pid)
    result['processes'][pid]=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
for path in ['raw_successor_features_v31/status.json','run_stable_observed_v30/status.json','run_stable_observed_v30/latest.COMMITTED.json']:
    p=r/path;result[path]=json.loads(p.read_text()) if p.exists() else None
for path in ['raw_successor_features_v31.log']:
    p=r/path;result[path]=p.read_text(errors='replace')[-5000:] if p.exists() else None
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
print(json.dumps(result))
(r/'raw_grounding_v31_setup.json').write_text(json.dumps(result,indent=2))
