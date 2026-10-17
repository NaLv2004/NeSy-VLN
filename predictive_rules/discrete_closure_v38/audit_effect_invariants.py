"""Learn generic action invariance from raw recorded outcomes, without names.

Only training layouts enter this audit. A held-out episode fold is used to
check the inferred invariants. There are no STOP/collision/location labels.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
records=sorted((ROOT/'raw_successor_features_v31').glob('counterfactual_*/*.json'))
counts=np.zeros((2,17),np.int64);changed=np.zeros_like(counts);maximum=np.zeros((2,17))
episodes=[set(),set()];source=[]
for meta in records:
    record=json.loads(meta.read_text());file=meta.with_suffix('.npz')
    if not file.exists():continue
    name=record['episode']
    assert 'P25_train_' in name
    fold=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%5==0
    episodes[int(fold)].add(name)
    with np.load(file,allow_pickle=False) as archive:
        mask=archive['probe_valid'].astype(bool)
        x=archive['features'][mask].astype(np.float32)
        future=archive['successor_features'][mask].astype(np.float32)
    diff=np.max(np.abs(future-x[:,None]),axis=(-1,-2))
    counts[int(fold)]+=len(diff);changed[int(fold)]+=(diff>1e-7).sum(0)
    maximum[int(fold)]=np.maximum(maximum[int(fold)],diff.max(0))
    source.append(dict(path=str(file),sha256=record['output_sha256']))
invariant=changed[0]==0
result=dict(time=time.time(),training_only=True,split='sha256(episode) mod 5: four fitting folds, one held-out train fold',
    fit_episodes=len(episodes[0]),heldout_episodes=len(episodes[1]),
    fit_positions=int(counts[0,0]),heldout_positions=int(counts[1,0]),
    fit_changed=changed[0].tolist(),heldout_changed=changed[1].tolist(),
    maximum_feature_change_fit=maximum[0].tolist(),maximum_feature_change_heldout=maximum[1].tolist(),
    induced_invariant_action_ids=np.where(invariant)[0].tolist(),
    heldout_invariant_violations=int(changed[1,invariant].sum()),
    raw_feature_identity_tolerance=1e-7,
    source_digest=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
    qualification='Empirical invariance across recorded training outcomes; not a universal physical proof. No rule was deployed by this audit.')
target=ROOT/'src_discrete_closure_v38/effect_invariants_audit.json'
target.write_text(json.dumps(result,indent=2));print(json.dumps(result))
