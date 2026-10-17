import json,time
from pathlib import Path
b=Path('/home/user/xjy/objectnav_qwen35_rl_20260921/runs');rows=[]
for p in b.rglob('*'):
 if p.is_file() and not p.is_symlink() and p.stat().st_size>30*2**20:
  rows.append({'path':str(p),'mib':round(p.stat().st_size/2**20,2),'links':p.stat().st_nlink,'mtime':p.stat().st_mtime})
sizes={};examples={}
for r in rows:
 name=Path(r['path']).name;sizes[name]=sizes.get(name,0)+r['mib'];examples.setdefault(name,[])
 if len(examples[name])<2:examples[name].append(r['path'])
result={'time':time.time(),'files':rows,'by_name_mib':sizes,'examples':examples}
Path('/home/user/xjy/nesy_vla_20261010/old_qwen_storage_v81.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'by_name_mib':sizes,'examples':examples},indent=2))
