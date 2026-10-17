"""Prepare a matched full-trajectory study, preserving original data isolation."""
import hashlib,json,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');prototype=base/'symbolic_paired_v12';root=base/'symbolic_execution_v38'
assert not root.exists(),'Inspect existing prepared study'
read=lambda p:json.loads(p.read_text())
integration=read(base/'neural_rule_training_integration_v38.json');assert integration['passed']
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
commit=read(source/'COMMITTED');assert hashlib.sha256((source/commit['checkpoint_file']).read_bytes()).hexdigest()==commit['checkpoint_sha256']
for name in ['nesy_vla','scripts']:
    shutil.copytree(prototype/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir()
template=read(base/'symbolic_recovery_v28/configs/recovery_mixture.json')
for name,enabled in [('conjunction_control',False),('neural_rule_weighting',True)]:
    cfg=json.loads(json.dumps(template))
    for key in ['initialize_from','initialize_symbolic_from','initialize_task_tokens_from','initialize_visual_from']:cfg.pop(key,None)
    cfg.update(root=str(root),architecture='discrete_execution_feedback_and_neural_conjunctions_v38',
        initialize_execution_from=str(source),execution_outcome_supervision=True,install_neural_conjunction_gates=True,
        execution_outcome_root='/home/user/xjy/strips_wm_navigation_20261009/data/search_cost_supervision_v3',
        demonstrations_per_rank=48,rollouts_per_rank=3,updates=40,snapshot_every=40,seed=2026101138,
        execution_positive_weight=read(base/'execution_learning_v34/dataset.json')['positive_weights'])
    cfg['program']['neural_conjunction_gates']=enabled;cfg['loss']['execution']=.5
    cfg.pop('rule_execution_cost',None);cfg['loss'].pop('rule_execution',None)
    assert cfg['horizon']==32 and cfg['revision_interval']==10 and cfg['evaluation_every']==10
    (root/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
record={'state':'prepared_not_started','time':time.time(),'source':str(source),'source_commit':commit,
    'initialization':'Same full source policy with fresh optimizers. Auxiliary and identity-initialized gate modules allocated in BOTH arms from the same seed.',
    'single_difference':'Activate neural per-rule scalar weighting in treatment; control executes original conjunction proof.',
    'full_trajectory_training':True,'successful_terminal_STOP_included':True,
    'demonstration_mixture':{'base':.5,'paired_goals':.25,'recovery':.25},
    'diagnostic_movement_only_weights_not_used':True,'gpus_per_arm':1,'total_gpus':2,
    'original_train_development_fixed_split_unchanged':True,'integration':integration,
    'no_direct_visual_action_module':True,'no_manually_defined_predicates':True,
    'original_development_and_fixed_evaluation_every':10,'navigation_improvement_proven':False,
    'launch_condition':'Review the completed matched v37 gate probe before launch; require free GPUs6/7 and sufficient disk.'}
(root/'experiment.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='integration'},indent=2))
