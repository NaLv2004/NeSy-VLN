import hashlib,json,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');prototype=base/'symbolic_paired_v12';root=base/'symbolic_recovery_v28'
assert not root.exists()
assert '4 passed' in (prototype/'trace_image_regression_results.log').read_text()
integration=json.loads((base/'recovery_integration_v27.json').read_text())
completion=json.loads((base/'symbolic_conjunctive_v23/completion_inspection.json').read_text())
assert not completion['pending']
assert all(j['status']['state']=='complete' and not j['live'] for j in completion['jobs'].values())
# Choose the parent by development results only, without consulting fixed80.
parent=max(completion['jobs'],key=lambda n:completion['jobs'][n]['best_score']['success_rate'])
source=base/'symbolic_conjunctive_v23/runs'/parent/'best_development'
commit=json.loads((source/'COMMITTED').read_text())
assert commit['optimizer_included']
assert hashlib.sha256((source/commit['checkpoint_file']).read_bytes()).hexdigest()==commit['checkpoint_sha256']
for name in ['nesy_vla','scripts']:
    shutil.copytree(prototype/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir()
template=json.loads((source.parent/'config.json').read_text())
for name,fraction in [('base_control',0.),('recovery_mixture',.25)]:
    cfg=json.loads(json.dumps(template));cfg.pop('initialize_task_tokens_from',None)
    cfg.update(root=str(root),architecture='causal_recovery_demonstrations_v28',initialize_from=str(source),
        demonstrations_per_rank=48,rollouts_per_rank=3,updates=60,snapshot_every=60,seed=20261011,
        teacher_decay=.995,demonstration_trace_every=10,rollout_trace_every=10,
        trace_image_store=str(base/'evidence_archives/live_png_store'),recovery_demo_fraction=fraction)
    if fraction:cfg['recovery_demonstration_root']=str(base/'data/recovery_demonstrations_v27')
    else:cfg.pop('recovery_demonstration_root',None)
    (root/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
record={'time':time.time(),'state':'prepared_not_started','source':str(source),'source_commit':commit,
    'selection':'Highest development success among the completed matched v23 arms; fixed80 excluded from parent selection.',
    'arms':{'base_control':{'base':.75,'paired':.25,'recovery':0.},
            'recovery_mixture':{'base':.5,'paired':.25,'recovery':.25}},
    'gpus_per_arm':3,'demonstrations_per_update':144,'recovery_complete':json.loads((base/'data/recovery_demonstrations_v27/COMPLETE').read_text()),
    'integration_audit':integration,'model_and_optimizer_initialization_identical':True,
    'random_prefixes':'Context and transition supervision only; never action imitation labels.',
    'no_new_actor_or_handwritten_predicates':True,'all_actions_from_discrete_generated_rules':True,
    'fixed_split_unchanged':True,'navigation_improvement_proven':False}
(root/'experiment.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'state':record['state'],'source':str(source),'source_commit':commit,'arms':record['arms']},indent=2))
