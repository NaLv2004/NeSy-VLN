import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
evidence=read(root/'evidence/rollout_proof_v8_startup.json')
assert evidence['process_identity_verified'] and evidence['committed_hash_verified']
assert evidence['previous_terminal_status']['state']=='training_budget_complete'
progress=read(root/'goal_progress.json');remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
progress['historical_visual_memory_best_development']=progress.get('current_best_development')
progress['historical_visual_memory_final_development']=read(root/'evidence/visual_memory_v6_final_dev.json')
progress['active_runs']['visual_memory_subgoal_v6'].update(last_observed_state='training_budget_complete',superseded=True,
    latest_preserved_update=2000,final_development=progress['historical_visual_memory_final_development'])
progress['active_runs']['rollout_proof_v8']=dict(path=remote+'/run_rollout_proof_v8',**evidence['launcher'],
    latest_status=evidence['status'],latest_checkpoint=evidence['checkpoint'],recent=evidence['recent'],
    process_identity_verified=True,committed_hash_verified=True,validation=evidence['validation'])
progress.update(remote_run=remote+'/run_rollout_proof_v8',remote_source=remote+'/src_rollout_proof_v8',
    architecture='anonymous_rollout_proof_subgoal_v8',last_verified_pid=evidence['status']['pid'],
    last_verified_update=evidence['status']['update'],last_observed_state=evidence['status']['state'],
    last_remote_check_unix=evidence['time'],next_regular_check_not_before_unix=evidence['time']+660,
    last_local_check_unix=(root/'evidence/rollout_proof_v8_startup.json').stat().st_mtime,
    current_best_development=None,latest_completed_development=None,last_record_update_unix=time.time())
progress['next_regular_local_check_not_before_unix']=progress['last_local_check_unix']+660
progress['definition_bound_trial']=dict(source_update=4000,head_updates=1000,
    sha256='774adf3f034c42e63f72b396bec9ae61096d3a5ee3c6b779b0561a4bdd05ce3b',
    first_action_accuracy=47/128,one_step_accuracy=22/128,teacher_path_endpoint_accuracy=72/128,
    development_successes=2,development_episodes=80,promoted=False,reason='No navigation gain despite rule-definition equivariance')
progress['rollout_proof_corrections']=dict(action_loss_future_image_invariant=True,
    state_dynamics_still_supervised_on_future_images=True,old_action_loss_not_invariant=True,
    exact_mixed_precision_transitions_compared=1280,deliberate_bf16_argmax_tie_fixed=True,
    inference_never_changed_to_use_observations_between_boundaries=True)
progress['state_aliasing_diagnosis']['v6_step1600']=dict(samples=128,conflicting_segments=80,conflicting_transitions=151,
    first_transition_correct=21,first_action_correct=58,development_successes=3,development_episodes=80)
progress['pending']=['Read v8 actual development outcomes after the stable polling gate; no duplicate checkpoint evaluations',
    'Confirm autonomous proof supervision improves real trajectories, not just surrogate loss',
    'Compare successful full inference with reference43/80 development and44/80 fixed; do not claim completion before both quality and speed are proven']
progress['current_turn_classification']='progress: evaluated v6 and rejected definition-bound pilot; isolated future-observation leakage into supervised action premises and an AMP row-lookup mismatch; corrected both and verified live v8 optimizer updates'
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
readme=(root/'README.md').read_text(encoding='utf-8')
section='''## Current implementation: autonomous proof supervision, v8

Source: `src_rollout_proof_v8`. Run: `run_rollout_proof_v8`. GPUs3/4/5 train;
GPU7 runs development80 once every400 updates. Source weights are the full
v6 step400 checkpoint, with a fresh optimizer, lr1e-4,640 windows/card and32
counterfactual states/card. The bounded run has2,000 updates and preserves
committed checkpoints. Other processes and the frozen splits are unchanged.

Two training/execution mismatches were corrected:

1. Previous action imitation used a state grounded from each actual future image
   as a planning premise. V8 derives every premise from the initial state and
   the generated rules, using the demonstrated action prefix for supervision.
   Actual future images still supervise world-model errors; they never correct
   these action-planning premises. A test changes every future image and verifies
   identical action-proof loss, while the dynamics loss changes. The old
   observed-premise action loss fails this invariance test.
2. AMP matrix multiplication of a one-hot state and a transition table could
   round distinct probabilities into a tie and change the successor ID. V8
   uses exact categorical row lookup in the forward pass with a straight-through
   gradient surrogate. A deliberate rounding counterexample and1,280 categorical
   transitions passed exact comparison against the executable lookup.

The runtime contract remains unchanged: complete generated definitions and
transition rules plus an unmodified discrete subgoal; complete exact proofs;
no maps, semantic hand labels, reward/value tables, actor fallback or neural
observation correction between revisions0/10/20/30.

V6 completed2,000 updates with best4/80 and final3/80 development success.
The definition-conditioned rule pilot trained1,000 updates on frozen grounding;
it passed state-renaming equivariance checks but reached only2/80, so it was not
promoted. V8 has verified real optimizer updates, not a navigation improvement
claim. The required reference parity remains unmet.

'''
heading='## Current implementation: visual-history subgoals, v6'
if '## Current implementation: autonomous proof supervision, v8' not in readme:
    readme=readme.replace(heading,section+'## Previous implementation: visual-history subgoals, v6 (completed)',1)
(root/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps(dict(update=evidence['status']['update'],recent_seconds=evidence['recent']['mean_seconds'],
    next_check_local=progress['next_regular_local_check_not_before_unix'],goal_complete=False)))
