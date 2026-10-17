import json
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
rows=[]
for name in ['counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5','counterfactual_proof_v8r1']:
    p=root/name;files=list(p.glob('*.npz'));m=json.loads((p/'manifest.json').read_text())
    rows.append(dict(path=str(p),files=len(files),bytes=sum(f.stat().st_size for f in files),
        training_only=m.get('training_only'),train_layouts=m['data']['train_layouts'],
        policy_checkpoint=m.get('policy_checkpoint'),episodes=len(m['episodes'])))
print(json.dumps(rows))
