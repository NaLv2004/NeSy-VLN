"""Persist measured CPU outcome, verified recovery and pending full experiment."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;V56=BASE/'definition_rules_v56'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
snapshot=read(V56/'last_observation.json');pending=read(HERE/'last_observation.json')
assert snapshot['supervisor']['state']=='running'
assert all(r['state']=='training' and r['update']>500 for r in snapshot['ranks'])
assert all(next(p for p in snapshot['processes'] if p['pid']==r['pid'])['command'] for r in snapshot['ranks'])
assert pending['handoff']['state']=='waiting_for_exact_predecessor_exit'
assert next(p for p in pending['processes'] if p['pid']==pending['handoff']['pid'])['command']
old=read(V56/'experiment_record.json')
old.update(phase='training',supervisor_pid=snapshot['supervisor']['pid'],rank_pids=[r['pid'] for r in snapshot['ranks']],
    latest_actual_update=min(r['update'] for r in snapshot['ranks']),last_remote_check_unix=snapshot['time'],
    observation=snapshot,retention_repair=read(HERE/'v56_retention_incident.json'))
old['source_files_sha256']['storage.py']=hashlib.sha256((V56/'storage.py').read_bytes()).hexdigest()
old['source_files_sha256']['build_sources.py']=hashlib.sha256((V56/'build_sources.py').read_bytes()).hexdigest()
write(V56/'experiment_record.json',old)
record=read(HERE/'experiment_record.json')
pilot=read(HERE/'pilot_result.json');audit=read(HERE/'grounding_audit.json')
record.update(phase='waiting_for_predecessor_terminal',pilot=pilot,grounding_audit=audit,
    contract=read(HERE/'contract.json'),observation=pending,handoff_pid=pending['handoff']['pid'],
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_goal_equivalence_v57',
    remote_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_goal_equivalence_v57',
    planned_training=dict(source='Best COMMITTED v56 checkpoint after all four full dev80 evaluations finish',
        gpus=[6,7],updates=2000,eval_every=500,development_episodes=80,max_actions=32,
        pair_loss_weight=1,representation_lr=2e-5,other_lr=1e-4,batch_per_gpu=128,all_action_positions_per_gpu=64),
    current_goal_turn_classification='progress',goal_status='active',completion_proven=False,
    next_action='After v56 terminal verify event-driven v57 startup. Assess unchanged full dev80 actual navigation, dynamic transition prediction, false stops and code diversity. Finalfixed80 remains untouched.',
    limitations=['CPU endpoint agreement is not physical navigation success.',
        'Source saw broader training data; pilot holdout is only disjoint from the additional fit.',
        'Action inverse accuracy fell98.2% to92.6% and hard-code perplexity38.5 to31.6; watch abstraction loss.',
        'No ablation separated continued fitting from added endpoint loss; no claim that every metric change is caused by this auxiliary.'])
record['source_files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
write(HERE/'experiment_record.json',record)
goal=read(BASE/'goal_progress.json')
goal.update(phase='training',supervisor_pid=old['supervisor_pid'],rank_pids=old['rank_pids'],
    rank_pid_coverage='Both actual GPU6/7 rank processes verified alive after full-state resume',
    latest_actual_update=old['latest_actual_update'],last_remote_check_unix=snapshot['time'],
    next_regular_check_not_before_unix=max(snapshot['time'],pending['time'])+630,
    current_goal_turn_classification='progress',previous_goal_turn_classification='progress',
    pending_successor=dict(record=str(HERE/'experiment_record.json'),remote_source=record['remote_source'],
        remote_run=record['remote_run'],handoff_pid=record['handoff_pid']),
    next_action=record['next_action'])
write(BASE/'goal_progress.json',goal)
for folder in (HERE,V56):
    for p in folder.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
print(json.dumps(dict(v56_update=goal['latest_actual_update'],v56_successes=snapshot['development'][-1]['successes'],
    v57_handoff_pid=record['handoff_pid'],next_check=goal['next_regular_check_not_before_unix'],goal_complete=False)))
