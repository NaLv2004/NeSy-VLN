"""Paired episode comparison of already completed scheduled validations."""
import json
import time
from pathlib import Path

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
paths={'v56_2000':ROOT/'run_definition_rules_v56/dev_002000',
       'v57_1000':ROOT/'run_goal_equivalence_v57/dev_001000'}
records={name:{r['episode']['episode_id']:r for r in
    [json.loads(p.read_text()) for p in folder.glob('*/trace.json')]} for name,folder in paths.items()}
assert all(len(r)==80 for r in records.values())
assert set(records['v56_2000'])==set(records['v57_1000'])
before=records['v56_2000'];after=records['v57_1000'];pairs=[]
for name in sorted(before):
    a=before[name];b=after[name]
    assert a['episode']==b['episode']
    pairs.append(dict(episode=name,domain=a['episode']['domain'],distance_group=a['episode']['distance_group'],
        before_success=bool(a['metrics'].get('success')),after_success=bool(b['metrics'].get('success')),
        before_steps=len(a['steps']),after_steps=len(b['steps']),
        before_initial_no_proof=bool(a.get('failed_proof',{}).get('horizon')==32),
        after_initial_no_proof=bool(b.get('failed_proof',{}).get('horizon')==32)))
def counts(rows):
    return dict(episodes=len(rows),both_success=sum(r['before_success'] and r['after_success'] for r in rows),
        gained=sum(not r['before_success'] and r['after_success'] for r in rows),
        lost=sum(r['before_success'] and not r['after_success'] for r in rows),
        both_failed=sum(not r['before_success'] and not r['after_success'] for r in rows),
        initial_no_proof_before=sum(r['before_initial_no_proof'] for r in rows),
        initial_no_proof_after=sum(r['after_initial_no_proof'] for r in rows))
result=dict(time=time.time(),counts=counts(pairs),by_group={
    field+':'+value:counts([r for r in pairs if r[field]==value])
    for field in ('domain','distance_group') for value in sorted({r[field] for r in pairs})},
    rows=pairs,new_rollouts=0,finalfixed80_used=False,
    limitation='Paired saved outcomes describe the change; continued optimization and the added terminal loss are not causally separated by this comparison.')
out=ROOT/'diagnostic_joint_endpoint_v59/saved_validation_comparison.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
