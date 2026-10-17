"""Build training-only discrete one-step world transitions, preserving splits."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.subgoal_data import TransitionBank,load_windows
from predictive_nesy.data import sha
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'discrete_visual_world_data_v15';assert not out.exists();out.mkdir()
source=root/'run_definition_world_v12/checkpoint_001600.pt';saved=torch.load(source,map_location='cpu',weights_only=False)
assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
before=[];after=[];actions=[];episodes=[];kinds=[];bank_reports=[]
for item in saved['manifest']['transition_banks']:
    bank=TransitionBank(item['path'],saved['manifest']['data']['train_layouts']);count=0
    for name,d in bank.episodes.items():
        for t in np.flatnonzero(d['probe_valid']):
            before.extend([d['codes'][t]]*17);after.extend(d['successors'][t]);actions.extend(range(17));episodes.extend([name]*17);kinds.extend([0]*17);count+=17
    bank_reports.append(dict(path=item['path'],manifest_sha256=sha(bank.directory/'manifest.json'),transitions=count))
windows,provenance=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
demo=0
for w in windows:
    for t in np.flatnonzero(w['action_valid']):
        before.append(w['targets'][t]);after.append(w['targets'][t+1]);actions.append(w['actions'][t]);episodes.append(w['episode_id']);kinds.append(1);demo+=1
before=np.asarray(before,dtype=np.uint8);after=np.asarray(after,dtype=np.uint8);actions=np.asarray(actions,dtype=np.int64);kinds=np.asarray(kinds,dtype=np.uint8)
held=np.asarray([int(hashlib.sha256(e.encode()).hexdigest()[:8],16)%5==0 for e in episodes]);episodes=np.asarray(episodes)
assert not(set(episodes[held])&set(episodes[~held]));assert before.shape==after.shape and before.shape[1]==49
assert before.max()<64 and after.max()<64 and actions.max()<17
target=out/'transitions.npz';np.savez_compressed(target,before=before,after=after,actions=actions,kind=kinds,held=held,episode=episodes)
manifest=dict(training_only=True,no_navigation_validation=True,source_checkpoint=str(source),source_sha256=sha(source),
    data=provenance,banks=bank_reports,frozen_vision=saved['manifest']['frozen_vision'],
    transitions=len(actions),demonstration_transitions=demo,fit=int((~held).sum()),held=int(held.sum()),
    episode_split='SHA256 episode modulo5; no episode shared by fit and internal holdout',
    field_allowlist=['before_discrete_visual_codes','after_discrete_visual_codes','action_id'],
    excluded=['map','pose','distance','depth','collision','reward','success','object_geometry'],
    unchanged_copy_baseline={split:float((before[mask]==after[mask]).mean()) for split,mask in [('fit',~held),('held',held)]},
    changed_patch_fraction=float((before!=after).mean()),npz_sha256=sha(target),created=time.time())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
