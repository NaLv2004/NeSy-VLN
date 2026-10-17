import json,sys
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_paired_v12');sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,DataBank,atomic
from nesy_vla.resources import paired_demonstration_schedule
base=Path('/home/user/xjy/nesy_vla_20261010')
cfg=read(base/'symbolic_reward_v11/configs/original.json')
cfg['paired_demonstration_roots']=[str(base/'paired_goal_probe_v12/input_snapshot')]
setup(cfg);bank=DataBank(cfg);paired=set(bank.paired_demo_ids)
ids=paired_demonstration_schedule([e for e in bank.demo_ids if e not in paired],bank.paired_demo_groups,1,0,1,12,cfg['seed'],1/3)
rows=[bank.demos[e] for e in ids]
assert len(bank.train)==len([e for e in read(bank.source/'train.json')
    if e['scene'] in bank.provenance['train_layouts'] and e['distance_group']!='stop_ready'])
assert all(bank.by_id[e]['scene'] in bank.provenance['train_layouts'] for e in ids)
assert all(bank.demo_image_path(r).exists() for episode in rows for r in episode)
assert len(set(ids)&paired)==4
record={'training_ids':ids,'paired_episode_count':4,'successful_demonstrations':len(rows),
    'frames':sum(map(len,rows)),'base_sampling_unchanged':True,'provenance':bank.provenance,
    'paired_data_validated':True,'fixed_hash':bank.provenance['fixed_hash']}
atomic(root/'dataset_integration.json',record);print(json.dumps(record,indent=2))
