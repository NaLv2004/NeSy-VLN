import json,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');prototype=base/'symbolic_paired_v12';root=base/'symbolic_teacher_v25'
assert not root.exists()
diagnoses=[json.loads((base/name).read_text()) for name in ['learning_objective_diagnosis_v24.json','learning_objective_synchronous_diagnosis_v24.json']]
assert diagnoses[0]['commit']==diagnoses[1]['commit'] and diagnoses[0]['episodes']==diagnoses[1]['episodes']
assert diagnoses[0]['mean_teacher_bit_disagreement']>.1 and diagnoses[1]['mean_teacher_bit_disagreement']==0
assert '110 passed' in (prototype/'teacher_regression_results.log').read_text()
for name in ['nesy_vla','scripts']:
    shutil.copytree(prototype/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir()
source=base/'symbolic_task_v15/runs/paired_tasks/latest'
template=json.loads((source.parent/'config.json').read_text())
for name,decay in [('ema_control',.995),('synchronous',0.)]:
    cfg=json.loads(json.dumps(template))
    cfg.pop('initialize_task_tokens_from',None)
    cfg.update(root=str(root),architecture='discrete_target_consistency_v25',initialize_from=str(source),
        teacher_decay=decay,demonstrations_per_rank=72,rollouts_per_rank=3,updates=40,snapshot_every=40,
        seed=20261011,demonstration_trace_every=10,rollout_trace_every=10)
    (root/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
record={'time':time.time(),'state':'prepared_not_started','source':str(source),'source_commit':diagnoses[0]['commit'],
    'arms':{'ema_control':.995,'synchronous':0.},'initial_model_and_optimizer_identical':True,
    'treatment':'The next-state target encoder uses the current learned symbol alphabet instead of an EMA lag. Targets remain detached hard tokens.',
    'only_intended_difference':'Target-encoder synchronization, including initial synchronization in the treatment.',
    'diagnostic_support':{'same_frame_symbol_disagreement':diagnoses[0]['mean_teacher_bit_disagreement'],
        'transition_loss_old_target':diagnoses[0]['terms']['transition'],'transition_loss_current_target':diagnoses[1]['terms']['transition']},
    'navigation_improvement_proven':False,'fixed_data_used_for_training':False,
    'all_actions_via_discrete_generated_rules':True,'no_manual_predicate_meanings':True}
(root/'experiment.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
