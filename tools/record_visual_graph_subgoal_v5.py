import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules';remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
evidence=json.loads((root/'evidence/visual_graph_v5_startup.json').read_text(encoding='utf-8-sig'))
assert evidence['process_identity_verified'] and evidence['committed_hash_verified']
progress=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
progress.update(remote_run=remote+'/run_visual_graph_subgoal_v5',remote_source=remote+'/src_visual_graph_subgoal_v5',
    architecture='anonymous_visual_graph_subgoal_v5',last_verified_pid=evidence['status']['pid'],last_verified_update=evidence['status']['update'],
    last_remote_check_unix=evidence['time'],next_regular_check_not_before_unix=evidence['time']+660,last_observed_state=evidence['status']['state'],
    last_record_update_unix=time.time(),current_best_development=None,latest_completed_development=None,
    contract_tests={'passed':13,'total':13},current_contract_tests={'passed':13,'total':13,'scope':'complete goal proofs, anonymous state grounding, future-action invariance, immutable rule execution, neural boundary gate'})
progress['active_runs']['graph_subgoal_v4'].update(last_observed_state='retired_world_model_inaccuracy',superseded=True,
    latest_preserved_update=3600,retired_checkpoint=evidence['previous_retired']['checkpoint'],
    best_development={'update':2400,'successes':4,'episodes':80},final_development=evidence['previous_final_development'])
progress['active_runs']['visual_graph_subgoal_v5']=dict(path=remote+'/run_visual_graph_subgoal_v5',**evidence['launcher'],
    latest_status=evidence['status'],latest_checkpoint=evidence['checkpoint'],recent=evidence['recent'],
    process_identity_verified=True,committed_hash_verified=True,validation=evidence['validation'])
progress['current_turn_classification']='progress: identified goal-label action-history dependence and weak true state transitions; implemented visual-only future labels and one-step supervised rule connections; launched verified replacement'
progress['pending']=['Measure actual navigation improvement from v5; no parity claim yet',
    'Keep immutable development and fixed layouts excluded from training',
    'Audit complete symbolic plans and neural isolation, then compare full inference latency and success against reference']
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
readme=(root/'README.md').read_text(encoding='utf-8').replace('## Current implementation: generated discrete subgoals, graph v4','## Previous implementation: generated discrete subgoals, graph v4 (retired)',1)
offset=readme.index('## Previous implementation:')
section='''## Current implementation: visually grounded discrete subgoals, v5

Active remote source: `src_visual_graph_subgoal_v5`; output:
`run_visual_graph_subgoal_v5`. Physical GPUs3/4/5 train and GPU7 evaluates.
All generated definitions, all128x17x128 rule parameters and the discrete
subgoal remain learned. Execution still accepts only exact goal-reaching
symbolic sequences, with neural revision only at actions0/10/20/30.

The prior v4 completed eight development evaluations up to step3,200, with a
best result of4/80. It was retired with the full step3,600 committed checkpoint
preserved. A training-only audit found only35/128 correct next-state codes.
Replacing generated goals by codes from the demonstrated future did not improve
first-action matching. Therefore the defect is not resolved by a goal-head
change alone.

V5 makes two connected changes to state supervision:

- A future state is grounded from the actual future image under the fixed
  boundary-generated definitions. Future action-history tags no longer enter
  the label encoder. Identical images under the same definitions produce the
  same state. STOP cannot create a new label merely because its ID was read.
  No semantic state or manually written STOP-transition rule is introduced.
- The full generated rule table is supervised on actual adjacent learned
  states and observed actions, in addition to open-loop multi-step propagation.
  This prevents one mistaken early transition from assigning conflicting labels
  to every later row. Future grounded states are training supervision only;
  execution never reads them or corrects its predicted state between revisions.

The complete goal-proof action objective remains, now supervised on the
demonstration's grounded state premises. The boundary program cannot see future
images. Definitions, transitions, subgoal and reconstruction heads initialize
fresh; only the existing visual encoder and boundary-history encoder are copied.
No reward/value table, semantic classifier or direct action head is present.

Exported rule arrays become read-only. Their digest is cached once at revision,
and each executed transition uses one table-row lookup. This removes redundant
full-program hashing on every action. End-to-end speed is still to be measured.

Thirteen tests passed, including future-action invariance, fixed-definition
image consistency, exact proof search, neural isolation and immutable rules.
The actual-data CUDA preflight had finite gradients and matched all16 valid
BF16/TF32 proof choices checked. Startup evidence is recorded in
`evidence/visual_graph_v5_startup.json` and `goal_progress.json`.

No v5 navigation improvement or reference parity has yet been established.
Development80 is used for selection; independent fixed80 remains untouched.

'''
readme=readme[:offset]+section+readme[offset:]
(root/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps({'recorded_update':evidence['status']['update'],'mean_seconds':evidence['recent']['mean_seconds'],'goal_complete':False}))
