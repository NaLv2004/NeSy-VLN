import json
from pathlib import Path

here=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
old_path=here.parent/'rule_planning_v32/progress.json'
old=read(old_path)
launch=read(here/'launcher.json')
startup=read(here/'startup.json')
replacement=read(here/'previous_replacement.json')
status=startup.get('status') or {}
record=dict(goal_status='active',goal_complete=False,
    previous_goal_turn_classification='verified wait: v32 PID verified live and existing scheduled results inspected',
    this_turn_classification='progress: trained-rule inverse likelihood and arbitrary causal branch boundaries implemented, v32 replaced without diagnostic rollouts, new formal trainer started',
    user_steering=old['user_steering'],requirements=old['requirements'],
    run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_action_grounded_v33',
    source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_action_grounded_v33',
    initialization=launch,
    changes=[
        'Inverse-action likelihood is evaluated from generated T probabilities at hard encoded successor IDs, with equivalent observed outcomes credited together',
        'Rule inverse loss directly trains transition rules and categorical successor representation; no runtime inverse head or actor fallback',
        'Counterfactual training chooses arbitrary causal program start0..t, using only the existing prefix before that start for program generation',
        'All17 successor images remain supervised label data only; the rule generator sees no future images',
    ],
    retained=old['changes'],training=dict(**old['training'],rule_inverse_weight=.5,
        counterfactual_program_start='Uniform integer0..t for each sampled recorded time t'),
    runtime=dict(pid=launch['pid'],gpu=7,process_alive_verified=startup['process_alive'],
        command=startup['command'],status=status.get('state'),latest_verified_update=status.get('update'),
        latest_commit=startup.get('latest_commit'),observed_at_remote_epoch=startup['observed_at'],
        next_routine_check_not_before_remote_epoch=startup['observed_at']+600,
        last_update_seconds=status.get('seconds'),memory_peak_gib=status.get('memory_peak_gib')),
    startup_evidence=dict(
        arbitrary_training_boundaries=startup['evidence_of_arbitrary_training_boundaries'],
        rule_inverse_values=startup['branch_rule_inverse_values']),
    collection=old['collection'],scheduled_development_results=[],
    conclusion='New training objective and runtime startup only; improvement unproven. Preceding v32 failed to improve. Original >=48/80 and >=5 executed actions/program requirements retained.',
    next_action='Do not poll before the recorded600s gate. Then inspect exact PID and COMMITTED checkpoint plus existing scheduled development results. No additional diagnostics or rollouts. Do not touch unrelated GPU0..5 processes.',
)
(here/'progress.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
old.update(superseded_by=record['run'],replacement_record=replacement,
    scheduled_development_results=replacement['completed_validation'],
    next_action='Do not restart v32. Follow ../action_grounded_v33/progress.json instead.')
old['runtime'].update(process=None,status='replaced',terminal_verified=True)
old_path.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(pid=launch['pid'],update=status.get('update'),state=status.get('state'),
    next_check=startup['observed_at']+600)))
