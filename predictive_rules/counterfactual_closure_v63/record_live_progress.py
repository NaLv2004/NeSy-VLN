import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
live=read(HERE/'last_observation.json');record=read(HERE/'experiment_record.json');ledger=read(BASE/'goal_progress.json')
assert live['status']['state']=='collecting',live['status']
alive={p['pid'] for p in live['processes'] if p['command']}
assert {live['launch']['pid'],*live['status']['workers']}<=alive
assert live['handoff']['pid'] in alive and live['training_supervisor']['state']=='waiting_collection'
physical=read(HERE/'sample_contract.json');numerical=read(HERE/'objective_contract.json')
completed=sum(s.get('completed',0) for s in live['ranks']);positions=sum(s.get('positions',0) for s in live['ranks'])
elapsed=live['time']-live['status']['started']
new_completed=sum(s.get('completed_since_restart',s.get('completed',0)) for s in live['ranks'])
rank_rates=[s.get('completed_since_restart',s.get('completed',0))/s['seconds']
            for s in live['ranks'] if s.get('seconds',0)>0]
rate=sum(rank_rates) if len(rank_rates)==len(live['ranks']) else 0
remaining=(600-completed)/rate if rate else None
record.update(remote_observation=live,state='collecting_with_training_handoff',
    completed_training_episodes=completed,collected_positions=positions,
    one_step_outcomes=positions*17,two_step_outcomes=positions*32,
    measured_collection_elapsed_seconds=elapsed,rough_remaining_collection_seconds=remaining,
    completed_since_restart=new_completed,measured_tasks_per_minute=rate*60,
    physical_example_audit=physical,objective_numerical_contract=numerical,
    training_handoff_pid=live['handoff']['pid'],training_has_started=False,
    scheduled_training_updates=2000,scheduled_full80_trials=1,
    scope='Navigation improvement remains unproven; these checks establish valid training data and objective computation only')
ledger.update(phase='collecting_measured_two_step_successors',main_training_started=False,
    remote_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_counterfactual_closure_v63',
    remote_future_training_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_counterfactual_closure_v63',
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_counterfactual_closure_v63',
    current_goal_turn_classification='progress',last_collection_check_unix=live['time'],
    next_collection_check_not_before_unix=live['time']+630,training_handoff_pid=live['handoff']['pid'],
    collection_coordinator_pid=live['launch']['pid'],collection_rank_pids=live['status']['workers'],
    next_action='After630s inspect the same v63 collector and handoff PIDs. Collection must finish600 train tasks, then verified data automatically starts2000updates fixed-abstraction rule fitting on GPUs6/7 and one original dev80 evaluation. Investigate only actual terminal failure or stale actual progress; preserve v60best27/80 and untouched finalfixed80.')
(HERE/'experiment_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(BASE/'goal_progress.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
print(json.dumps(dict(completed=completed,total=600,positions=positions,one_step_outcomes=positions*17,
    two_step_outcomes=positions*32,rough_minutes_remaining=remaining/60 if remaining else None,
    training='waiting_collection',next_check=live['time']+630)))
