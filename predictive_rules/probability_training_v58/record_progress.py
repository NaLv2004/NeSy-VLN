"""Record verified lifecycle and independent experiment; never infer success."""
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
V57=BASE/'goal_equivalence_v57'
V56=BASE/'definition_rules_v56'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

snapshot=read(V57/'last_observation.json')
handoff=read(V57/'handoff_result.json')
development=[json.loads(line) for line in (V56/'development.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
assert handoff['state']=='training_dispatched'
assert handoff['predecessor_terminal']['exit_code']==0
assert {r['update'] for r in development}=={500,1000,1500,2000}
assert all(r['episodes']==80 for r in development)
assert snapshot['manifest']['initialize_sha256']==handoff['source']['sha256']
assert all(next(p for p in snapshot['processes'] if p['pid']==r['pid'])['command'] for r in snapshot['ranks'])
old=read(V56/'experiment_record.json')
old.update(phase='finished',terminal=handoff['predecessor_terminal'],development=development,
    best_commit=handoff['source'],final_successes=development[-1]['successes'],
    best_successes=max(r['successes'] for r in development),
    completion_proven=False,successor=str(V57/'experiment_record.json'))
write(V56/'experiment_record.json',old)
record=read(V57/'experiment_record.json')
record.update(phase=snapshot['ranks'][0]['state'],observation=snapshot,handoff=handoff,
    actual_initialization=handoff['source'],last_remote_check_unix=snapshot['time'],
    next_regular_check_not_before_unix=snapshot['time']+630,
    supervisor_pid=snapshot['supervisor']['pid'],rank_pids=[r['pid'] for r in snapshot['ranks']],
    latest_actual_update=min(r['update'] for r in snapshot['ranks']),
    task_specificity_audit=str(V57/'task_specificity_audit.json'),
    recorded_dynamics_audit=str(V57/'recorded_dynamics_audit.json'),
    finished_backup_retention=str(V57/'finished_backup_retention.json'),
    independent_probability_experiment=str(HERE/'experiment_record.json'),
    current_goal_turn_classification='progress',completion_proven=False,
    next_action='Read scheduled full dev80 results after the 630-second gate. Do not repeat checkpoint evaluations. The two-step marginal-likelihood pilot did not improve held-out prediction and must not be promoted.')
write(V57/'experiment_record.json',record)
pilot=dict(phase='running',source=handoff['source'],remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_probability_training_v58',
    remote_diagnostic='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diagnostic_probability_training_v58',
    reference='https://proceedings.mlr.press/v97/hafner19a.html',
    adaptation='Two-step exact categorical marginal likelihood, versus detached modal-path likelihood; identical frozen v56 representation, initial rule weights, minibatches and learning rates.',
    live_training_modified=False,dev80_used=False,finalfixed80_used=False,device='CPU',threads=2,
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
    goal_status='active',completion_proven=False,
    limitations=['No navigation result from this diagnostic.',
        'Source representations previously saw this entire training corpus; holdout is disjoint only for additional rule fitting.',
        'PlaNet reports model-dependent overshooting benefits; this idea is a hypothesis, not a guarantee.'])
if (HERE/'result.json').exists():
    result=read(HERE/'result.json')
    initial=result['history'][0]['metrics']['all_branches']['holdout']
    final=result['history'][-1]['metrics']['all_branches']
    control=result['history'][-1]['metrics']['modal_control']['holdout']
    pilot.update(phase='complete_not_promoted',result=result,
        decision='Do not deploy. Training likelihood improves, but held-out two-step accuracy and NLL deteriorate relative to the unchanged source. Small improvement over a similarly overfitted modal control does not establish improvement.',
        key_results=dict(initial_holdout_accuracy=initial['two_step_probability_accuracy'],
            final_holdout_accuracy=final['holdout']['two_step_probability_accuracy'],
            final_fit_accuracy=final['fit']['two_step_probability_accuracy'],
            final_control_holdout_accuracy=control['two_step_probability_accuracy'],
            initial_holdout_nll=initial['two_step_probability_nll'],
            final_holdout_nll=final['holdout']['two_step_probability_nll']))
write(HERE/'experiment_record.json',pilot)
goal=read(BASE/'goal_progress.json')
goal.pop('handoff_pid',None)
goal.pop('pending_successor',None)
goal.update(authoritative_record=str(V57/'experiment_record.json'),
    remote_run=record['remote_run'],remote_source=record['remote_source'],phase=record['phase'],
    supervisor_pid=record['supervisor_pid'],rank_pids=record['rank_pids'],
    latest_actual_update=record['latest_actual_update'],last_remote_check_unix=snapshot['time'],
    next_regular_check_not_before_unix=snapshot['time']+630,
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    rank_pid_coverage='Both actual GPU6/7 rank command lines verified; scheduled full dev80 at500 in progress at observation time.',
    independent_experiment=str(HERE/'experiment_record.json'),next_action=record['next_action'],
    previous_history=str(V56/'experiment_record.json'),goal_status='active',completion_proven=False)
write(BASE/'goal_progress.json',goal)
print(json.dumps(dict(v56_final=development[-1]['successes'],v57_update=record['latest_actual_update'],
    v57_state=record['phase'],next_check=snapshot['time']+630,pilot_phase=pilot['phase'],goal_complete=False)))
