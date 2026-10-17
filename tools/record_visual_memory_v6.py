import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
check=read(root/'evidence/visual_memory_v6_startup.json')
assert check['process_identity_verified'] and check['committed_hash_verified'] and check['status']['update']>0
progress=read(root/'goal_progress.json');remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
previous=progress['active_runs']['visual_graph_subgoal_v5']
previous.update(last_observed_state='training_budget_complete',superseded=True,latest_preserved_update=6000,
    best_development={'update':4000,'successes':5,'episodes':80},final_development={'update':6000,'successes':3,'episodes':80})
progress['historical_visual_graph_best_development']=progress['current_best_development']
progress['historical_visual_graph_final_development']=progress['latest_completed_development']
progress.update(remote_run=remote+'/run_visual_memory_subgoal_v6',remote_source=remote+'/src_visual_memory_subgoal_v6',
    architecture='anonymous_visual_memory_subgoal_v6',last_observed_state=check['status']['state'],
    last_verified_pid=check['status']['pid'],last_verified_update=check['status']['update'],last_remote_check_unix=check['time'],
    last_local_check_unix=(root/'evidence/visual_memory_v6_startup.json').stat().st_mtime,
    current_best_development=None,latest_completed_development=None,last_record_update_unix=time.time())
progress['next_regular_local_check_not_before_unix']=progress['last_local_check_unix']+660
progress['next_regular_check_not_before_unix']=check['time']+660
progress['active_runs']['visual_memory_subgoal_v6']=dict(path=progress['remote_run'],**check['launcher'],
    latest_status=check['status'],latest_checkpoint=check['checkpoint'],process_identity_verified=True,
    committed_hash_verified=True,recent=check['recent'],validation=check['validation'])
progress['fixed_grounding_trial']=dict(source_update=4000,updates=600,checkpoint_sha256='a3ba2e440353be959f59045a6e6a7e9f6bb9d3a684138fed6c6bf45ec01d3cbb',
    frozen_parameters_bitwise_unchanged=True,first_action_before=53/128,first_action_after=63/128,
    first_transition_before=17/128,first_transition_after=21/128,development_successes=3,development_episodes=80,promoted=False)
progress['state_aliasing_diagnosis']=dict(training_only=True,checkpoint_update=4000,samples=128,
    segments_with_conflicting_next_state=89,conflicts=159,
    definition='same generated program, current discrete state and action have distinct observed next-state labels',
    reference_visual_only_alphabet_conflicting_segments=77,
    future_action_tags_rejected=True)
progress['current_contract_tests']=dict(passed=8,total=8,devices=['cpu','cuda'],
    scope='v6 function-preserving initialization, temporal image sensitivity, repeated-image invariance, no future action tags, export isolation, finite demonstration and counterfactual gradients, neural boundary rejection')
progress['contract_tests']={'passed':8,'total':8}
progress['pending']=['Read v6 development results no sooner than the stable polling gate; do not duplicate evaluation',
    'Audit whether visual history reduces conflicting state-transition labels and improves actual navigation',
    'Preserve fixed80 for final comparison; success parity and full navigation speed remain unproved']
progress['current_turn_classification']='progress: completed v5; rejected a bounded fixed-grounding trial; identified contradictory deterministic state labels; implemented stutter-invariant visual-history grounding and verified live v6 optimizer updates'
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
readme=(root/'README.md').read_text(encoding='utf-8')
section='''## Current implementation: visual-history subgoals, v6

Source: `src_visual_memory_subgoal_v6`. Output: `run_visual_memory_subgoal_v6`.
GPUs3/4/5 train, GPU7 evaluates; other processes are preserved. The 2,000-update
run initializes from the full v5 step4,000 weights (fresh optimizer), with a
zero-output learned visual-memory extension. The initial policy is preserved.
Batch size640/card gives1,920 windows/update plus32 transition probes/card.
Development80 is evaluated once every400 updates. Final fixed80 remains isolated.

The network still generates every definition, every128x17x128 transition
parameter and its unmodified categorical subgoal. The interpreter derives a
complete exact goal-reaching action sequence before execution. It uses only
immutable arrays between revisions0/10/20/30, without image/state correction.
There are no reward/value tables, semantic states, collision heads or actor fallback.

Why this change: v5 completed6,000 updates with best5/80 and final3/80 on
development. In128 training segments,89 contained contradictory successors for
the same state/action pair under the same generated program. Freezing grounding
and fitting only rules improved first-action matching from53/128 to63/128, but
development reached only3/80; that bounded candidate was not promoted.

V6 learns a history representation from visual feature differences. It never
uses future action tags to define training-label states. Repeating exactly the
same visual codes leaves the memory unchanged, so STOP or repeated frames cannot
manufacture a new goal state. The memory participates in neural grounding only
at program-revision boundaries. For training, future observations provide labels
under the same generated definitions; for execution, these observations and
memory tensors are absent from the exported rule program.

Eight targeted checks passed on CPU and CUDA. A real640-window mixed-precision
batch plus32 transition probes had finite gradients and13.58GiB peak allocated
memory. Confirmed live startup evidence is in `evidence/visual_memory_v6_startup.json`.
These tests establish implementation constraints, not navigation improvement.
Success parity with the reference43/80 development and44/80 fixed is unproved.

The single-frame CUDA-graph vision optimization is used in the v6 evaluator.
Its exact-token equivalence was measured on8,232 cached training tokens. The
earlier2.24x compute-only replay speedup was measured for v5; full v6 navigation
throughput must be measured independently.

'''
heading='## Current implementation: visually grounded discrete subgoals, v5'
if '## Current implementation: visual-history subgoals, v6' not in readme:
    readme=readme.replace(heading,section+'## Previous implementation: visually grounded discrete subgoals, v5 (completed)',1)
(root/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps(dict(recorded_update=check['status']['update'],seconds=check['recent']['mean_seconds'],
    next_regular_local_check_not_before=progress['next_regular_local_check_not_before_unix'],goal_complete=False)))
