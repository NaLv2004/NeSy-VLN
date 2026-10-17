import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;BASE=HERE.parent;V57=BASE/'goal_equivalence_v57'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
observation=read(V57/'last_observation.json')
source=next(r for r in observation['development'] if r['update']==1000)
record=dict(phase='cpu_pilot_running',source_commit=dict(update=1000,
    sha256='58abc7f6a46587f2312412b1848c1e84dc64a374ba725b81f51dd8c0bca880bf'),
    source_development=source,exec_session_id=41488,
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_joint_endpoint_v59',
    remote_diagnostic='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diagnostic_joint_endpoint_v59',
    device='CPU',threads=2,updates=96,
    adaptation='Compare matched continuing v57 training with and without successful terminal-memory pair agreement. No inference change, no goal substitution, no manually defined symbol meanings.',
    finalfixed80_used=False,navigation_rollouts=0,main_training_unchanged=True,
    source_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    goal_status='active',completion_proven=False)
for name in ('result.json','saved_validation_comparison.json'):
    if (HERE/name).exists():record[name.split('.')[0]]=read(HERE/name)
if 'result' in record:
    record['phase']='cpu_pilot_complete_not_promoted'
    record.pop('exec_session_id',None)
    record['decision']='Do not promote. At96 updates held-out joint terminal agreement96.875% vs control100%; STOP-goal preimage96.875% vs100%, initial joint-goal alias1.5625% vs0. No improvement in autonomous memory rollout. Fitting losses alone do not justify deployment.'
write(HERE/'experiment_record.json',record)
current=read(V57/'experiment_record.json')
current.update(phase=observation['supervisor']['state'],observation=observation,
    latest_actual_update=min(r['update'] for r in observation['ranks']),
    best_successes=max(r['successes'] for r in observation['development']),
    last_remote_check_unix=observation['time'],next_regular_check_not_before_unix=observation['time']+630,
    independent_experiment=str(HERE/'experiment_record.json'),
    saved_goal_audit=str(BASE/'probability_training_v58/goal_feasibility_v57.json'),
    next_action='v57 completed2000; best1000 remains26/80, final16/80. Reject v59 memory-agreement pilot. Collect fresh training-only all-action successors with best v57 policy before further dynamics training. No goal replacement or neural action fallback.')
write(V57/'experiment_record.json',current)
goal=read(BASE/'goal_progress.json')
goal.update(phase=current['phase'],latest_actual_update=current['latest_actual_update'],
    last_remote_check_unix=observation['time'],next_regular_check_not_before_unix=observation['time']+630,
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    independent_experiment=str(HERE/'experiment_record.json'),next_action=current['next_action'],
    latest_verified_full80=dict(update=1000,successes=source['successes'],episodes=80),
    rank_pid_coverage='Supervisor exited0 and exact supervisor/rank PIDs are absent; GPU6/7 idle. Stale rank evaluating statuses are superseded by terminal supervisor and actual process evidence.',
    goal_status='active',completion_proven=False)
write(BASE/'goal_progress.json',goal)
print(json.dumps(dict(main_update=current['latest_actual_update'],best_full80=current['best_successes'],
    next_check=observation['time']+630,pilot_phase=record['phase'],goal_complete=False)))
