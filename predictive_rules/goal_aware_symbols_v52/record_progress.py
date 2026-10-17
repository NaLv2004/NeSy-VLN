"""Keep the original goal and complete evidence, including negative results."""
import json
from pathlib import Path
here=Path(__file__).parent;root=here.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
launch=read(here/'launcher.json');snapshot=read(here/'last_observation.json')
stable=read(root/'stable_probability_v50/report.json');path=read(root/'path_rollout_v51/report.json')
prior=read(root/'contextual_symbols_v48/experiment_record.json')
prior.update(status='finished_not_improved',final_commit=launch['predecessor_final'],
    final_development=launch['predecessor_development'],successor='goal_aware_symbols_v52',
    next_action='Do not restart; preserve final3000 and best500. Follow goal_aware_symbols_v52/experiment_record.json.')
(root/'contextual_symbols_v48/experiment_record.json').write_text(json.dumps(prior,ensure_ascii=False,indent=2),encoding='utf-8')
record=dict(goal_status='active',completion_proven=False,
    previous_goal_turn_classification='progress: restarted monitor service and verified fresh remote data',
    this_goal_turn_classification='progress: repaired and high-precision-verified probability arithmetic, completed94root and24task diagnostics, identified no physical success improvement, implemented and launched task-conditioned symbolic abstractions after predecessor completion',
    objective='Strictly greater than80% on unchanged complete dev80 and independent finalfixed80; all executed actions symbolically derived',
    completion_audit=dict(dev80_threshold=65,finalfixed80_threshold=65,new_dev80_result=None,
        finalfixed80_used=False,goal_achieved=False,remaining='Demonstrate effective discrete prediction and >80% independent navigation success'),
    invariants=prior['invariants'],
    completed_diagnostics=dict(
        arithmetic=stable['arithmetic_contract'],
        matched94roots=stable['root_summary'],
        matched24training_tasks=dict(original=stable['baseline'],stable_expected=stable['stable'],maximum_likelihood_path=path['maximum_path']),
        conclusion='Numerical bug is real but correcting it alone did not improve1/24 physical success. Highest-path-likelihood also1/24, so it is not promoted.'),
    research=[dict(url='https://proceedings.mlr.press/v119/nair20a.html',
        use='Inspired task-conditioned anonymous abstraction; this implementation still exports full symbolic transition tables and runs a non-neural planner. This is an adaptation, not a claimed reproduction.'),
        dict(url='https://arxiv.org/abs/2505.22772',use='Motivates preserving factual transition supervision rather than replacing it by uncalibrated planning-score supervision; no critic/value network adopted.')],
    model_changes=[
        'New zero-initialized learned task embedding enters boundary context before full anonymous definition/rule generation.',
        'Current, historical and recorded future state labels use the same task-conditioned definition set, frozen until renewal.',
        'Different task IDs can define different abstractions of the same scene; all IDs are learned, with no manually named predicates.',
        'Unchanged categorical32 memory and categorical64 observation. Memory grounds real history only at renewal, minimum5actions; O refreshes each action.',
        'Stable normalized float64 success/failure Bellman planning over full generated integer tables; no learned action/value/reward head at runtime.'],
    initialization=launch['source_commit'],training=launch['training'],planned_updates=3000,
    eval_every=500,dev_episodes=80,maximum_actions=32,
    contracts=launch['contract'],predecessor_final=launch['predecessor_final'],
    predecessor_development=launch['predecessor_development'],runtime=snapshot,
    dashboard='http://127.0.0.1:8893/',
    next_regular_check_not_before_unix=snapshot['time']+600,
    next_action='At or after the600second gate inspect the exact live v52 process IDs, commits and scheduled full dev80. Use existing saved traces to diagnose whether goal-conditioned symbols improve real prediction, not just training accuracy. Do not rerun completed v49/v50/v51 diagnostics or reopen finalfixed80.')
(here/'experiment_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
current=dict(goal_status='active',completion_proven=False,objective=record['objective'],
    authoritative_record=str(here/'experiment_record.json'),remote_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_goal_aware_symbols_v52',
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_goal_aware_symbols_v52',
    invariants=record['invariants'],last_remote_check_unix=snapshot['time'],next_regular_check_not_before_unix=snapshot['time']+600,
    minimum_stable_poll_interval_seconds=600,last_verified_update=snapshot['ranks'][0]['update'],
    verified_processes=snapshot['processes'],previous_history=str(root/'contextual_symbols_v48/experiment_record.json'),next_action=record['next_action'])
(root/'goal_progress.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(update=current['last_verified_update'],next_poll=current['next_regular_check_not_before_unix'],
    best_previous_v48=max(r['successes'] for r in launch['predecessor_development']),goal_achieved=False,
    gradient_norm=snapshot['ranks'][0].get('gradient_norm'),task_basis_separation=snapshot['ranks'][0].get('task_basis_separation'))))
