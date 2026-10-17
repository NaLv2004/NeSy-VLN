"""Read-only attribution of storage growth and reward configuration."""
import json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');old=Path('/home/user/xjy/strips_wm_navigation_20261009')
r={'time':time.time(),'disk':dict(zip(['total','used','free'],shutil.disk_usage(b)))}
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=100);return {'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-1000:]}
r['owned_root_sizes']=cmd(['du','-x','-B1','--max-depth=1','/home/user/xjy'])
r['nesy_sizes']=cmd(['du','-x','-B1','--max-depth=2',str(b)])
r['strips_sizes']=cmd(['du','-x','-B1','--max-depth=2',str(old)])
r['current_reward_config']=json.loads((b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json').read_text())
r['training_and_inference_processes']=cmd(['ps','-eo','pid,lstart,args'])
(b/'storage_reward_inspection_v81.json').write_text(json.dumps(r,indent=2))
out={k:v for k,v in r.items() if k not in ['current_reward_config','training_and_inference_processes']}
for key in ['owned_root_sizes','nesy_sizes','strips_sizes']:
 rows=[]
 for line in r[key]['stdout'].splitlines():
  fields=line.split('\t',1)
  if len(fields)==2:rows.append({'gib':round(int(fields[0])/2**30,3),'path':fields[1]})
 out[key]=sorted(rows,key=lambda x:x['gib'],reverse=True)[:32]
out['reward_config']={k:v for k,v in r['current_reward_config'].items() if 'reward' in k or 'env' in k}
print(json.dumps(out,indent=2))
