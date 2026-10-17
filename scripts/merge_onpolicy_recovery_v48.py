"""Audit and merge only successful train-only corrective demonstrations."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'data/onpolicy_recovery_v48';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(out/'manifest.json');expected={r['id']:r for r in manifest['items']};records=[];frames=[];failed=[];prefix_checked=0
if (out/'COMPLETE').exists():raise SystemExit('Already merged, preserve existing commit')
for rank in range(6):
 done=read(out/f'rank_{rank}/COMPLETE');assert done['completed']==done['attempts']==len(manifest['items'][rank::6])
 for r in done['records']:
  folder=out/f'rank_{rank}'/r['id'];assert r==read(folder/'COMMITTED');item=expected[r['id']]
  for file,key in [('examples.json','examples_sha256'),('audit.json','audit_sha256'),('execution_outcomes.json','outcomes_sha256')]:assert sha(folder/file)==r[key]
  audit=read(folder/'audit.json');assert audit['item']==item and audit['success']==r['success'];assert item['episode']['scene'] in manifest['train_layouts'] and item['episode']['scene'] not in manifest['excluded_layouts']
  rows=read(folder/'examples.json');original=read(Path(item['source_rollout']));assert sha(Path(item['source_rollout']))==item['source_rollout_sha256']
  if not r['success']:assert not rows;failed.append(r);continue
  assert len(rows)<=32 and rows[-1]['action']==0 and audit['trace'][-1]['success']
  history=[];actions=[]
  for i,row in enumerate(rows):
   image=(out/row['image']).resolve();assert image.is_relative_to(out.resolve()) and sha(image)==row['image_sha256']
   pixel=hashlib.sha256(np.asarray(Image.open(image).convert('RGB')).tobytes()).hexdigest();assert pixel==row['pixel_sha256']
   history.append(row['image']);assert row['history']==history and row['previous_actions']==actions
   assert row['supervision_mask']==(i>=item['prefix_length']) and row['action']==audit['trace'][i]['action']
   if i<=item['prefix_length'] and i<len(original['steps']):assert pixel==original['steps'][i]['image_sha256'];prefix_checked+=1
   actions.append(row['action']);frames.append({'episode_id':item['id'],'source_episode_id':item['episode']['episode_id'],'category':row['category'],'path':str(image),'label':row['action'],'supervised':row['supervision_mask'],'image_sha256':row['image_sha256']})
  records.append(r)
assert len(records)+len(failed)==len(expected)
(out/'training_rows.json').write_text(json.dumps(frames,indent=2))
result={'time':time.time(),'attempts':len(expected),'successful':len(records),'failed_attempts':len(failed),'frames':len(frames),'supervised_frames':sum(r['supervised'] for r in frames),'context_frames':sum(not r['supervised'] for r in frames),'verified_original_prefix_frames':prefix_checked,'training_rows_sha256':sha(out/'training_rows.json'),'manifest_sha256':sha(out/'manifest.json'),'source_data_hashes':manifest['source_data_hashes'],'no_heldout_corrections':True,'failures_excluded_from_imitation':True,'source_rl_model_success_rate':None,'qualification':'Success count is the private training teacher recovery yield, not learned navigation success.'}
(out/'COMPLETE').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
