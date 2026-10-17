"""Full discrete-program training, with train-only paired counterfactual goals."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');source=base/'symbolic_paired_v12';root=base/'symbolic_task_v15'
assert not root.exists(),'Inspect existing run instead of launching a duplicate'
assert shutil.disk_usage(base).free>4*2**30,'Storage must be repaired before starting'
for name in ['nesy_vla','scripts']:
    shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir();out=root/'runs/paired_tasks';out.mkdir(parents=True)
cfg=json.loads((base/'symbolic_reward_v11/configs/original.json').read_text())
source_checkpoint=cfg.pop('initialize_from')
reference=Path(cfg['reference_root']);split=json.loads((reference/'research/split_lock.json').read_text())
environment=json.loads((reference/'configs/experiment.json').read_text());dataset=Path(environment['source_data'])
for name in ['train.json','fixed80.json']:
    assert hashlib.sha256((dataset/name).read_bytes()).hexdigest()==split['hashes'][name]
allowed=set(split['train_scenes'])-set(split['development_scenes'])
tasks=json.loads((dataset/'train.json').read_text())
vocabulary=sorted({'Find the '+e['category']+' and stop close to it.' for e in tasks if e['scene'] in allowed})
cfg.update(root=str(root),architecture='task_token_paired_full_discrete_v15',generator_backend='task_tokens',
    task_vocabulary=vocabulary,initialize_task_tokens_from=source_checkpoint,
    paired_demonstration_roots=[str(base/'data/paired_goals_v12')],paired_demo_fraction=.25,
    training_mode='supervised',updates=60,rollouts_per_rank=2,demonstrations_per_rank=24,
    training_chunk_batch=24,maximum_chunk_batch=48,training_frame_budget=220,revision_chunk_limit=24,
    learning_rate=.0002,vision_learning_rate=.00001,evaluation_every=10,snapshot_every=60,
    demonstration_trace_every=10,rollout_trace_every=10,canonical_vision_batch=8)
assert cfg['horizon']==32 and cfg['revision_interval']==10 and cfg['discrete_reasoning']
for key in ['training_reward_overrides','initialize_symbolic_from','initialize_visual_from']:cfg.pop(key,None)
config=root/'configs/paired_tasks.json';config.write_text(json.dumps(cfg,indent=2))
record={'time':time.time(),'source_checkpoint':source_checkpoint,'gpus':[0,1,2,3,4,5],
    'train_only_task_vocabulary':vocabulary,'optimizer':'fresh; added task embeddings',
    'all_actions_through_generated_rules':True,'no_manual_predicate_semantics':True,
    'purpose':'Learn goal-conditioned symbols/rules on complete successful trajectories and matched counterfactual goals.',
    'evaluation':'Existing isolated development80 and fixed80 every10 updates; never included in training.'}
(root/'experiment.json').write_text(json.dumps(record,indent=2))
command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','scripts/supervise.py',
    '--config',str(config),'--out',str(out),'--gpus','0,1,2,3,4,5']
with (out/'supervisor.log').open('ab') as log:
    child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                           start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
record.update(supervisor_pid=child.pid,command=command);(root/'process.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
