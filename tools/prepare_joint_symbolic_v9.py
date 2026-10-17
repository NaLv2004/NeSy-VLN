import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_joint_v9';source=base/'joint_symbolic_v9_code'
assert '52 passed' in (source/'verification.log').read_text()
assert not root.exists(),'Inspect an existing experiment, never overwrite'
root.mkdir()
for name in ['nesy_vla','scripts','tests']:shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(source/'verification.log',root/'verification.log');(root/'configs').mkdir()
origin=base/'symbolic_robust_v7/runs/robust_multieffect/best_development'
marker=json.loads((origin/'COMMITTED').read_text());target=root/'initialization_source';target.mkdir()
file=origin/marker['checkpoint_file'];os.link(file,target/file.name)
assert marker['optimizer_included'] and hashlib.sha256((target/file.name).read_bytes()).hexdigest()==marker['checkpoint_sha256']
(target/'COMMITTED').write_text(json.dumps(marker,indent=2))
cfg=json.loads((base/'symbolic_robust_v7/configs/robust_multieffect.json').read_text())
cfg.update(root=str(root),architecture='joint_discrete_inverse_v9',initialize_from=str(target),training_mode='joint',
    rollouts_per_rank=8,demonstrations_per_rank=12,training_chunk_batch=24,maximum_chunk_batch=64,training_frame_budget=180,revision_chunk_limit=18)
(root/'configs/joint.json').write_text(json.dumps(cfg,indent=2))
small=json.loads(json.dumps(cfg));small.update(updates=1,rollouts_per_rank=1,demonstrations_per_rank=1,
    training_chunk_batch=2,maximum_chunk_batch=2,training_frame_budget=20,revision_chunk_limit=2,candidate_every=1)
(root/'configs/integration.json').write_text(json.dumps(small,indent=2))
score=json.loads((origin.parent/'best_score.json').read_text())
manifest={'time':time.time(),'source':str(origin),'source_checkpoint':marker,'source_selected_by':'development80 success only; fixed80 not used for source selection',
    'source_development':score,'changes':['joint PPO and successful demonstrations','photometric augmentation only on expert rows','inverse action objective uses hard next-state codes','equivalent batched rule gates'],
    'unchanged':['automatically learned predicates','all actions via generated soft logic','max32 actions','revision every10 actions','original isolated validation','original environment rewards'],
    'neural_image_action_bypass':False,'handwritten_predicates':False,'maps':False,'tests_passed':52,'new_success_unproven':True}
(root/'experiment.json').write_text(json.dumps(manifest,indent=2))
wrapper=root/'run_integration.py';wrapper.write_text("import runpy,sys,torch\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root))\ntorch.cuda.set_per_process_memory_fraction(.30)\nsys.argv=['scripts/train.py','--config','configs/integration.json','--out',str(root/'integration')]\nrunpy.run_path(str(root/'scripts/train.py'),run_name='__main__')\n")
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
with (root/'integration.log').open('ab') as log:
    child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(wrapper)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':child.pid,'time':time.time(),'out':str(root/'integration'),'gpu':6,'memory_fraction':.30}
(root/'integration_process.json').write_text(json.dumps(record,indent=2));print(json.dumps({'integration':record,'source':marker,'source_dev_successes':score['successes']}))
