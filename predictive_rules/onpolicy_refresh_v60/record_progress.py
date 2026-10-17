import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
obs=read(HERE/'last_observation.json')
record=dict(phase=obs['status'].get('state','starting'),observation=obs,
    source=dict(architecture='goal_equivalence_v57',update=1000,successes=26,episodes=80,
        sha256='58abc7f6a46587f2312412b1848c1e84dc64a374ba725b81f51dd8c0bca880bf'),
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_onpolicy_refresh_v60',
    remote_training_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_onpolicy_refresh_v60_training',
    remote_collection='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/onpolicy_successors_v60',
    remote_future_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_onpolicy_refresh_v60',
    main_training_started=False,finalfixed80_used=False,validation_used_for_collection=False,
    hypothesis='Current policy visits states unlike v52 policy used in old v54 successor cache. Refresh all-action measured transitions on training layouts; no changes to symbolic planning.',
    prior_aggregation_rounds=['onpolicy_successors_v40','onpolicy_successors_v54'],
    research_reference='https://icml.cc/2012/papers/833.pdf',
    research_scope='Iterative model learning under the current planner, inspired by Ross/Bagnell2012. This implementation does not claim their policy-error guarantee applies to learned partial-observation abstractions.',
    next_action='At630-second gate inspect the original collector2184984, ranks2184985/2184986 and event-wait handoff2186519. Handoff automatically audits durable training-only files and real-gradient/symbolic-planner contract, then starts1000-update v60 on GPUs6/7. Verify exact supervisor/rank identities if dispatched; never duplicate a live job or rerun old evaluations.',
    source_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
    last_remote_check_unix=obs['time'],next_regular_check_not_before_unix=obs['time']+630,
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    goal_status='active',completion_proven=False)
for name in ('checkpoint_retention_applied.json','table_retention_applied.json'):
    if (HERE/name).exists():record[name.split('.')[0]]=read(HERE/name)
if (HERE/'handoff_verified.json').exists():record['handoff_verification']=read(HERE/'handoff_verified.json')
record['prepared_training']=dict(updates=1000,eval_every=500,development_episodes=80,max_actions=32,
    unchanged_architecture=True,unchanged_loss_weights=True,unchanged_learning_rates=True,
    current_policy_branch_fraction=.5,older_cumulative_branch_fraction=.5,
    memory_grounding_only_at_rule_renewal=True,minimum_renewal_actions=5,
    neural_action_fallback=False,source_preparation_is_not_training_start=True)
write(HERE/'experiment_record.json',record)
goal=read(BASE/'goal_progress.json')
goal.update(authoritative_record=str(HERE/'experiment_record.json'),phase=record['phase'],
    remote_source=record['remote_source'],remote_run=record['remote_collection'],
    remote_future_training_run=record['remote_future_run'],main_training_started=False,
    supervisor_pid=obs['status'].get('pid'),rank_pids=obs['status'].get('workers',[]),
    latest_actual_update=None,last_completed_training_update=2000,
    last_remote_check_unix=obs['time'],next_regular_check_not_before_unix=obs['time']+630,
    next_action=record['next_action'],rank_pid_coverage='Exact coordinator and both collector command lines observed live; subsequent checks must inspect these handles, not assume progress from status files.',
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    goal_status='active',completion_proven=False,
    latest_verified_full80=dict(update=2000,successes=16,episodes=80),
    best_verified_full80=dict(run='goal_equivalence_v57',update=1000,successes=26,episodes=80))
write(BASE/'goal_progress.json',goal)
print(json.dumps(dict(phase=record['phase'],collector_pids=goal['rank_pids'],next_check=obs['time']+630,goal_complete=False)))
