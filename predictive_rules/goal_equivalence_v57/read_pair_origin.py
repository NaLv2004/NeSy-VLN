import json
from pathlib import Path
p=Path('/home/user/xjy/nesy_vla_20261010/data/paired_goals_v12/pairs/P25_train_100_chair_near_1_062')
for folder in [p,p/'chair',p/'table']:
 print('FOLDER',str(folder),[v.name for v in folder.iterdir() if v.suffix=='.json'])
 for f in folder.glob('*.json'):
  d=json.loads(f.read_text()); print(f.name,json.dumps(d)[:9000])