"""Record authoritative experiment history without claiming goal completion."""
import json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
root=HERE.parent
launch=json.loads((HERE/'launcher.json').read_text(encoding='utf-8'))
snapshot=json.loads((HERE/'last_observation.json').read_text(encoding='utf-8'))
v46=json.loads((root/'discounted_rules_v46/report.json').read_text(encoding='utf-8'))
v47=json.loads((root/'qwen_features_v47/report.json').read_text(encoding='utf-8'))
terminal=json.loads((root/'boundary_balanced_v43/terminal_compatibility_report.json').read_text(encoding='utf-8'))
record=dict(goal_achieved=False,goal_status='active',
    previous_goal_turn_classification='progress: repaired dashboard and verified live remote telemetry',
    current_goal_turn_classification='progress: completed immutable discounted-rule and Qwen feature probes; implemented contextual discrete observation definitions, verified causal/gradient/proof/archive contracts, drained owned old run with committed checkpoint and launched real two-GPU training',
    objective='Strictly greater than80% success on unchanged complete development80 and independent finalfixed80, all motor actions symbolically derived',
    invariants=dict(no_map=True,no_manual_semantic_state_definitions=True,no_neural_action_fallback=True,
        memory_grounding_only_at_rule_renewal=True,observation_refresh_per_action=True,
        minimum_actions_between_renewals=5,max_actions=32,train_layouts_unchanged=True,
        finalfixed80_used=False,gpus=[6,7],unrelated_gpu_processes_untouched=True),
    completed_diagnostics=dict(
        teacher_terminal_compatibility=dict(training_trajectories=terminal['episodes'],metrics=terminal['metrics'],
            conclusion='Most recorded successful teacher routes end in the generated goal; this does not establish autonomous goal/planning quality.'),
        discounting=dict(results=v46['results'],promoted=False,
            conclusion='1/24 to2/24 matched training tasks is weak evidence; observation accuracy stays about22%. No dev80 retest.'),
        qwen_original_features=dict(results={name:{mode:{'accuracy':v['heldout']['accuracy'],'macro_accuracy':v['heldout']['macro_accuracy']} for mode,v in modes.items()} for name,modes in v47['results'].items()},
            frames=v47['frame_count'],promoted=False,limitations=v47['limitations'],
            conclusion='No improvement in this fixed diagnostic; keep current frozen ConvNeXt. No prior RL Qwen adapters or validation examples used.')),
    changes=[
        'Generate complete64x128 anonymous observation definition vectors from actual history at each rule boundary.',
        'Freeze definitions and full exported rules between renewals; encode current/future training observations with the same definitions.',
        'Condition auxiliary raw-feature reconstruction on these definitions, without exposing it to the runtime planner.',
        'Re-ground discrete memory only at renewal; ordinary memory advancement remains from exported current-state/current-observation/action rules.',
        'At a renewal, score the preceding observation prediction against observations encoded with the old definitions before switching.',
        'Preserve unchanged NumPy all-outcome symbolic Bellman derivation and exact generated goal; no state reward table or direct action head.'],
    predecessor_final=launch['predecessor_final'],predecessor_best=launch['predecessor_best'],
    predecessor_development=[{k:r[k] for k in ('update','successes','observation_prediction_accuracy','collision_action_rate')} for r in launch['predecessor_development']],
    initialization=launch['source_commit'],optimizer='Fresh AdamW, representation2e-5, rule/goal/new definitions1e-4',
    planned_updates=3000,evaluation_every=500,development_episodes=80,
    per_gpu_batch=dict(demonstrations=128,all_action_branch_positions=64),
    contract=launch['contracts'],runtime=snapshot,
    next_normal_poll_not_before=snapshot['time']+600,
    next_action='Wait for scheduled complete dev80 evidence; inspect exact saved PID and commit after600seconds. Diagnose failures from saved traces, not substitute a simpler validation set. Do not restart v43 or repeat completed v46/v47 probes.')
(HERE/'experiment_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
old_path=root/'boundary_balanced_v43/experiment_record.json'
old=json.loads(old_path.read_text(encoding='utf-8'))
old.update(status='superseded_after_verified_graceful_drain',final_commit=launch['predecessor_final'],
    best_commit=launch['predecessor_best'],final_development=record['predecessor_development'],
    successor='contextual_symbols_v48',next_action='Do not restart. Follow contextual_symbols_v48/experiment_record.json.')
old_path.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
current_path=root/'goal_progress.json'
history=root/'goal_progress_before_v48.json'
if current_path.exists() and not history.exists():history.write_bytes(current_path.read_bytes())
current=dict(goal_status='active',completion_proven=False,objective=record['objective'],
    authoritative_record=str(HERE/'experiment_record.json'),remote_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_contextual_symbols_v48',
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_contextual_symbols_v48',
    invariants=record['invariants'],last_remote_check_unix=snapshot['time'],
    next_regular_check_not_before_unix=snapshot['time']+600,minimum_stable_poll_interval_seconds=600,
    last_verified_update=snapshot['ranks'][0]['update'],verified_processes=snapshot['processes'],
    previous_history=str(history),next_action=record['next_action'])
current_path.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(record=str(HERE/'experiment_record.json'),update=snapshot['ranks'][0]['update'],next_check=snapshot['time']+600)))
