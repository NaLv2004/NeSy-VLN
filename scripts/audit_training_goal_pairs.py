"""Check whether existing supervision demands different actions for identical RGB."""
import hashlib,sys,time,json,collections
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_joint_v9');sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,DataBank,atomic
cfg=read(root/'configs/joint.json');setup(cfg);bank=DataBank(cfg);start=time.time();groups={}
for rows in bank.demos.values():
    for row in rows:
        digest=hashlib.sha256(bank.demo_image_path(row).read_bytes()).hexdigest()
        groups.setdefault(digest,[]).append({'episode':row['episode_id'],'category':row['category'],'action':row['action'],'image':row['image']})
cross=[rows for rows in groups.values() if len(set(r['category'] for r in rows))>1]
disagree=[rows for rows in cross if len(set(r['action'] for r in rows))>1]
result={'time':time.time(),'seconds':time.time()-start,'training_only':True,'frames':sum(len(x) for x in groups.values()),
    'unique_png_sha256':len(groups),'cross_goal_identical_images':len(cross),'cross_goal_with_different_action_labels':len(disagree),
    'example_groups':disagree[:10],'qualification':'Exact encoded PNG equality. Different bytes do not prove different scenes or semantic content.'}
atomic(root/'training_goal_pair_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='example_groups'},indent=2))
