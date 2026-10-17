import json,time
from pathlib import Path
HERE=Path(__file__).parent;ROOT=HERE.parent
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
old=read(ROOT/'goal_aware_symbols_v52/experiment_record.json')
snapshot=read(ROOT/'goal_aware_symbols_v52/last_observation.json')
old.update(status='finished_not_breakthrough',runtime=snapshot,
    final_development=snapshot['development'],final_commit=snapshot['latest'],best_commit=snapshot['best'],
    successor='hybrid_replay_v54',next_action='Do not restart terminal v52; preserve best2000 and final3000, follow new data collection and hybrid replay.')
old['completion_audit'].update(new_dev80_result=max(r['successes'] for r in snapshot['development']),goal_achieved=False)
(ROOT/'goal_aware_symbols_v52/experiment_record.json').write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
probe=read(ROOT/'rl_perception_v53/report.json');baseline=read(HERE/'collected_baseline.json')
record=dict(goal_status='active',completion_proven=False,
    previous_goal_turn_classification='progress: restarted local dashboard and verified live remote telemetry',
    current_goal_turn_classification='progress: completed task-prefixed RL-Qwen feature probe and runtime frontend audit; finished v52 full evaluation; identified runtime/teacher-input mismatch; implemented and gradient-checked hybrid replay; launched two-card current-policy training-only transition aggregation and dependent training handoff',
    objective=old['objective'],invariants=old['invariants'],
    completion_audit=dict(dev_required=65,dev_episodes=80,finalfixed_required=65,finalfixed_episodes=80,
        recent_v52_best=11,new_v54_validation=None,finalfixed80_used=False,goal_achieved=False),
    predecessor=dict(run='run_goal_aware_symbols_v52',terminal_supervisor=snapshot['supervisor'],
        terminal_processes=snapshot['processes'],best=snapshot['best'],final=snapshot['latest'],
        development=[dict(update=r['update'],successes=r['successes'],episodes=r['episodes']) for r in snapshot['development']]),
    completed_diagnostics=dict(rl_qwen_perception=probe,
        frontend_equivalence=read(ROOT/'rl_perception_v53/frontend_equivalence.json'),
        current_policy_baseline=baseline,
        interpretation='Runtime observation-code prediction remains poor on newly visited TRAINING states (about22%); offline substituting grounded memory improves it only to25%, so memory mismatch is real but cannot explain all errors. Initial prediction19% also shows visual/state-coverage limitations. Qwen feature swap was not promoted.'),
    changes=['Add supervised factual transition loss on predicted-memory plus current-observed-code inputs, matching runtime.',
        'During replay, memory is grounded once at the boundary and then predicted before each action; future observations are targets only.',
        'Keep old autonomous-observation rollout, teacher-forced branches, inverse/reconstruction and goal supervision.',
        'Aggregate120 balanced training-only trajectories under the v52best2000 symbolic policy, with all17 physical action outcomes.',
        'Fresh-policy branches use recorded renewal times; half branch samples are fresh, half preserve old demonstrations/probe coverage.',
        'No model parameters, runtime action head, handcrafted semantic/collision states, reward/value tables, or maps added.'],
    contract=read(HERE/'contract.json'),collection=read(HERE/'collection_launcher.json'),
    verified_collection=dict(coordinator_pid=2127870,rank_pids=[2127871,2127872],last_direct_startup_check=1792193287,
        startup_completed=16,latest_independent_baseline_committed_episodes=baseline['episodes']),
    verified_handoff=dict(pid=2130014,state='waiting_for_specific_collector',verified_unix=1792193636),
    planned_training=dict(gpus=[6,7],run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_hybrid_replay_v54',
        source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_hybrid_replay_v54',
        updates=2000,eval_every=500,eval_tasks=80,max_actions=32,batch_per_rank=128,branch_batch_per_rank=64,
        representation_lr=2e-5,other_lr=1e-4,optimizer='fresh AdamW; source weights exactly v52best2000'),
    next_regular_check_not_before_unix=1792193887,
    disk_retention=read(ROOT/'rl_perception_v53/retention.json'),
    references=[dict(url='https://arxiv.org/abs/1612.06018',
        application='Inspired training on model-induced intermediate premises and current-policy data. This is an adaptation, not a reproduction of H-DAgger-MC or its theorem.'),
        dict(url='https://github.com/etalvitie/hdaggermc',application='Primary implementation reference for model-induced replay; no reward/value model adopted.')],
    dashboard='http://127.0.0.1:8893/',
    next_action='Inspect exact collector/handoff handles after600s gate or at the handoff startup verification. Confirm120complete and new training actual updates. Do not repeat completed feature probes or run finalfixed80. If v54 has begun, verify first committed weights, fresh branch provenance, finite hybrid loss and scheduled full dev80.')
(HERE/'experiment_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
current=dict(goal_status='active',completion_proven=False,objective=record['objective'],invariants=record['invariants'],
    authoritative_record=str(HERE/'experiment_record.json'),remote_run=record['planned_training']['run'],
    remote_source=record['planned_training']['source'],phase='current_policy_collection_then_hybrid_training',
    collector_pid=2127870,collector_rank_pids=[2127871,2127872],handoff_pid=2130014,
    last_remote_check_unix=1792193287,next_regular_check_not_before_unix=1792193887,
    minimum_stable_poll_interval_seconds=600,previous_history=str(ROOT/'goal_aware_symbols_v52/experiment_record.json'),
    next_action=record['next_action'])
(ROOT/'goal_progress.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(goal_achieved=False,v52_final=snapshot['latest']['update'],collection_pid=2127870,
    handoff_pid=2130014,next_poll=current['next_regular_check_not_before_unix'])))
