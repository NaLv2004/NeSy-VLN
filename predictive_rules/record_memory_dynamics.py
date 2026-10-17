import json
from pathlib import Path
BASE=Path(__file__).parent
observation=json.loads((BASE/'memory_dynamics_observation.json').read_text())
startup=json.loads((BASE/'v77_v79_startup.json').read_text())
results={n:json.loads((BASE/n/'result.json').read_text()) for n in ('expanded_dynamics_v71','stable_rules_v75','expanded_history_v76')}
record=dict(goal_complete=False,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    completed=results,trace_audit=observation['v71_trace_audit'],startup=startup,
    repair=dict(run='expanded_history_v76',fault='Generated evaluator appended to its prepared list while iterating, never submitted first environment action',
        fixed='Single preparation loop and per-episode derivation/audit/execution loop; generation replacement limited to first occurrence',
        same_sealed_predictor=True,retraining=False,repaired_successes=47,static_iteration_regression_passed=True),
    new_experiments=dict(memory_dynamics_v77='Source50, fixed visual/definitions/grounding/goal; train next-M and next-O rules',
        joint_memory_dynamics_v78='Source50, additionally fit generic visual residual',
        full_history_v79='Source50, full1200 compact-history residual; completed1000 but failed predeclared heldout gate, no navigation evaluation'),
    next='Inspect actual completion of v77/v78 no earlier than recorded next_regular_check; compare candidate on same80 only if heldout gate passes. Keep source50 and44. No repeated evaluation. Investigate remaining memory/O aliasing with train-only evidence.',
    last_remote_check_unix=startup['time'],next_regular_check_not_before_unix=startup['time']+600)
dest=BASE/'memory_dynamics_v77/experiment_record.json';dest.write_text(json.dumps(record,indent=2))
p=BASE/'goal_progress.json';state=json.loads(p.read_text())
state.update(goal_status='active',completion_proven=False,authoritative_record=str(dest),
    phase='v71_verified50_v76_repaired47_v77_v78_future_memory_training',
    current_goal_turn_classification='progress',previous_goal_turn_classification='progress',
    previous_goal_turn_evidence='Code-only repository publication completed; preceding experiment turn launched v75/v76 and GPU acceleration.',
    last_remote_check_unix=startup['time'],next_regular_check_not_before_unix=startup['time']+600,
    next_action=record['next'],remote_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_memory_dynamics_v77',
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_memory_dynamics_v77',
    supervisor_pid=2306869,rank_pids=[2306940,2306941],collection_coordinator_pid=None,collection_rank_pids=[],
    active_parallel_runs=[dict(name=n,**startup['runs'][n]['supervisor.json']) for n in ('memory_dynamics_v77','joint_memory_dynamics_v78')],
    best_verified_current_curriculum=dict(run='expanded_dynamics_v71',update=500,successes=50,episodes=80,success_rate=.625,
        checkpoint='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_expanded_dynamics_v71/checkpoint_000500.pt',
        sha256=results['expanded_dynamics_v71']['chosen']['sha256'],trace_audit=observation['v71_trace_audit']),
    latest_verified_current_curriculum=dict(run='expanded_history_v76',update=500,successes=47,episodes=80),
    latest_actual_update=startup['runs']['memory_dynamics_v77']['status_rank0.json']['update'])
p.write_text(json.dumps(state,indent=2))
print(json.dumps(dict(best=state['best_verified_current_curriculum'],next_regular_check_not_before_unix=state['next_regular_check_not_before_unix'])))
