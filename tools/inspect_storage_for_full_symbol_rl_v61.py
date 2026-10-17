import json,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010')
lines=subprocess.check_output(['du','-x','-B1','--max-depth=2',str(b)],text=True).splitlines()
sizes=sorted([(int(line.split('\t',1)[0]),line.split('\t',1)[1]) for line in lines],reverse=True)
result={'time':time.time(),'free_gib':shutil.disk_usage(b).free/2**30,'largest':[{'path':p,'gib':s/2**30} for s,p in sizes[:25]]}
result['candidate_checkpoints']=[]
for marker in (b/'symbolic_execution_v38').rglob('COMMITTED'):
 if marker.parent.name not in ['step_000020','step_000030']:continue
 m=json.loads(marker.read_text());p=marker.parent/m.get('checkpoint_file','checkpoint.pt')
 result['candidate_checkpoints'].append({'folder':str(marker.parent),'commit':m,'exists':p.exists(),'symlink':p.is_symlink(),'links':p.stat().st_nlink if p.exists() else None,'gib':p.stat().st_size/2**30 if p.exists() else None})
(b/'storage_scope_v61.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
