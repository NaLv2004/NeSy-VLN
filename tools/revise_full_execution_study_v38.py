"""Reject ineffective action gates; test measured-outcome grounding on full histories."""
import hashlib,json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_execution_v38';read=lambda p:json.loads(p.read_text())
assert not (root/'processes.json').exists()
probe=read(base/'neural_conjunction_learning_v37/result.json');assert probe['state']=='complete'
arms=probe['arms'];control=arms['conjunction_control']['learning'][-1]['holdout'];gated=arms['neural_rule_weighting']['learning'][-1]['holdout']
assert control['expert_action_accuracy']==gated['expert_action_accuracy'] and gated['chosen_move_observed_collision_rate']>control['chosen_move_observed_collision_rate']
review={'time':time.time(),'decision':'Do not promote neural action gates. Test execution-outcome supervision on full trajectories instead.',
    'probe_result':str(base/'neural_conjunction_learning_v37/result.json'),
    'static_control':control,'static_neural_weighting':gated,
    'gates_disabled':arms['neural_rule_weighting']['same_weights_gates_disabled'],
    'limitation':'Movement-only learning omits STOP and history; final diagnostic weights are not deployed.',
    'next_comparison':'Same full causal training; auxiliary predictions detached from predicates in control and connected in treatment.'}
(root/'probe_review.json').write_text(json.dumps(review,indent=2))
old=read(root/'experiment.json');(root/'rejected_gate_preparation.json').write_text(json.dumps(old,indent=2))
# Parent selected using development only, after complete v28 runs.
parents=[base/'symbolic_conjunctive_v23/runs/conjunctive',base/'symbolic_recovery_v28/runs/base_control',base/'symbolic_recovery_v28/runs/recovery_mixture']
assert all(read(p/'status.json')['state']=='complete' for p in parents)
parent=max(parents,key=lambda p:read(p/'best_score.json')['success_rate']);source=parent/'best_development'
commit=read(source/'COMMITTED');assert hashlib.sha256((source/commit['checkpoint_file']).read_bytes()).hexdigest()==commit['checkpoint_sha256']
template=read(root/'configs/conjunction_control.json')
for name,detach in [('detached_outcome_control',True),('grounded_outcome_feedback',False)]:
    cfg=json.loads(json.dumps(template));cfg.update(architecture='full_trajectory_discrete_execution_supervision_v38',
        initialize_execution_from=str(source),install_neural_conjunction_gates=False,detach_execution_symbols=detach)
    cfg['program']['neural_conjunction_gates']=False
    (root/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
old.update(state='prepared_not_started',time=time.time(),source=str(source),source_commit=commit,
    selection='Highest development success of completed v23/v28 parents. Fixed80 excluded from selection.',
    initialization='Identical full source policy and same auxiliary head initialization, fresh optimizers.',
    single_difference='Only whether auxiliary execution-outcome loss backpropagates into learned predicates and definitions.',
    neural_action_gates_enabled=False,gpus_per_arm=3,total_gpus=6,
    arms=['detached_outcome_control','grounded_outcome_feedback'],probe_review=review,
    launch_condition='Completed v28 jobs; six free GPUs1..6 and sufficient disk.')
(root/'experiment.json').write_text(json.dumps(old,indent=2))
print(json.dumps({'source':str(source),'source_commit':commit,'decision':review['decision'],'arms':old['arms']},indent=2))
