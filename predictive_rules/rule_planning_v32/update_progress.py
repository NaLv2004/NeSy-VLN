"""Record verified evidence from a completed, read-only remote observation."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
prior_path = HERE.parent / 'raw_grounding_v31/progress.json'
prior = json.loads(prior_path.read_text(encoding='utf-8'))
replacement = json.loads((prior_path.parent / 'replacement_record.json').read_text(encoding='utf-8'))
launch = json.loads((HERE / 'launcher.json').read_text(encoding='utf-8'))
observed = json.loads((HERE / 'monitor_latest.json').read_text(encoding='utf-8'))
status = observed['status']
record = dict(
    goal_status='active', goal_complete=False,
    previous_goal_turn_classification='progress: implemented hard-forward backward-planning gradients into generated rules and launched v32',
    this_turn_classification='verified wait: exact trainer PID is live with 1464 updates; training-only feature collection completed; inspected existing scheduled results and installed rate-limited monitor',
    user_steering=prior['user_steering'], requirements=prior['requirements'],
    run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_rule_planning_v32',
    source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_rule_planning_v32',
    initialization=launch,
    changes=[
        'Add teacher-action loss through complete generated transition tables and exact hard backward proof',
        'Argmax transitions and rounded log-confidence costs unchanged in forward execution; only training derivatives use a zero-valued straight-through surrogate',
        'Rule probabilities, definition generator and goal generator receive the new planning objective; no action head or continuous-state runtime bypass',
        'All 8066 recorded training branch positions now have unquantized successor visual features, across all17 actions',
    ],
    training=dict(demonstration_batch=32, branch_positions_batch=64,
        branch_outcomes_batch=1088, full_table_planning_batch=4, planning_loss_weight=.25,
        representation_lr=1e-5, rule_goal_lr=1e-4, ema_decay=.995,
        planned_updates=6000, scheduled_evaluation_every=500, maximum_actions=32),
    runtime=dict(pid=launch['pid'], gpu=7, process=observed['processes'].get(str(launch['pid'])),
        latest_verified_update=status.get('update'), status=status.get('state'),
        latest_commit=observed['latest_commit'], last_update_seconds=status.get('seconds'),
        memory_peak_gib=status.get('memory_peak_gib'), observed_at_remote_epoch=observed['observed_at'],
        next_routine_check_not_before_remote_epoch=observed['observed_at']+600),
    collection=dict(**observed['collector_status'], process=observed['processes'].get('1843119'),
        committed_positions=observed['raw_positions'], total_action_outcomes=observed['raw_positions']*17,
        note='Complete and PID absent; do not restart collection. GPU6 released.'),
    scheduled_development_results=observed['validation'],
    training_window_metrics=observed['training_windows'],
    conclusion='No demonstrated success-rate improvement. Scheduled step500=14/80,step1000=13/80; frequency requirement satisfied at both. Preserve original development80 and >=48 successes target.',
    next_action='Continue existing formal v32 training. No extra diagnostic tests or comparison rollouts. Next routine inspection after the recorded 600-second gate; inspect exact PID, COMMITTED checkpoint and scheduled results. Do not touch unrelated GPU0..5 jobs.',
)
(HERE / 'progress.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
prior.update(superseded_by=record['run'], runtime_terminal_verified=True,
    replacement_record=str(prior_path.parent / 'replacement_record.json'),
    replacement_summary=replacement,
    next_action='Do not restart v31. Follow ../rule_planning_v32/progress.json and its exact recorded process instead.')
prior['runtime']['process_alive_verified']=False
prior['runtime']['terminal_state']='superseded; saved checkpoints preserved'
prior_path.write_text(json.dumps(prior,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(progress=str(HERE/'progress.json'), update=status.get('update'),
    successes=[x['successes'] for x in observed['validation']],
    next_check=observed['observed_at']+600)))
