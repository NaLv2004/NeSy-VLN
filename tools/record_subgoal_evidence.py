import json,time
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'predictive_rules/goal_progress.json';d=json.loads(path.read_text())
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
d.update(last_verified_pid=514113,last_verified_update=4312,last_observed_state='training',
    last_remote_check_unix=1792087409.8479428,next_regular_check_not_before_unix=1792088010,
    contract_tests=dict(passed=8,total=8),current_contract_tests=dict(passed=8,total=8,scope='subgoal schemas, exact rule proof, data boundaries and neural isolation'))
d['active_runs']['subgoal_v1'].update(state='training',last_observed_update=4312,actual_optimizer_updates_confirmed=True,
    last_update_seconds=.2127082347869873,global_batch_size=320,memory_peak_gib=9.553778171539307,
    latest_checkpoint=dict(update=4300,sha256='21d09c5b87a446567008d3e45c11d8928833420af5949b9fcb158ca3181733ef'))
d['current_best_development']=dict(run='subgoal_v1',update=1600,successes=1,episodes=80)
d['latest_completed_development']=dict(run='subgoal_v1',update=4000,successes=0,episodes=80,
    exact_plan_found_rate=.4838709677419355,boundary_gate_violations=0,all_executed_actions_have_exact_subgoal_proof=True)
d['subgoal_training_diagnosis']=dict(training_only=True,checkpoint_update=1200,episodes=92,found_plans=47,
    short_stop_proofs=26,matched_actual_successor_encodings=1,
    interpretation='Symbolic proof is valid under generated rules, but learned dynamics and subgoal grounding are inaccurate')
d['rule_inverse_trial']=dict(path=root+'/subgoal_rule_inverse_trial',gpu=4,updates=400,
    candidate_sha256='e267816e24341c250785ba4ea9d89958c9f0f8dc40a0f986ba5cb508dfcee914',
    training_diagnostic=dict(episodes=92,found_plans=29,short_stop_proofs=20,matched_actual_successor_encodings=1),
    success_improvement_proven=False,development_evaluation='running',local_source_revision='v2')
d['pending']=['Read the bounded rule-inverse trial navigation result; do not promote based only on inverse accuracy',
    'Inspect completed subgoal training without confusing symbolic reachability with navigation success',
    'Improve learned dynamics and subgoal grounding without adding return tables, semantic priors or between-boundary observations',
    'Reference parity and speed requirements remain unachieved']
d['last_record_update_unix']=time.time();path.write_text(json.dumps(d,indent=2),encoding='utf-8')
