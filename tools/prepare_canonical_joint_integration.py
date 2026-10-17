import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_joint_v9';source=base/'joint_symbolic_v9_code'
record=json.loads((root/'integration_process.json').read_text())
assert not Path(f"/proc/{record['pid']}/cmdline").exists()
assert json.loads((root/'integration/status.json').read_text())['state']=='complete'
assert '57 passed' in (source/'verification.log').read_text()
snapshot=root/'source_before_canonical_replay';snapshot.mkdir()
for name in ['nesy_vla','tests','configs']:shutil.copytree(root/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['nesy_vla','tests']:shutil.copytree(source/name,root/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for path in (root/'configs').glob('*.json'):
    cfg=json.loads(path.read_text());cfg.update(canonical_vision_batch=8,require_exact_ppo_replay=True)
    path.write_text(json.dumps(cfg,indent=2))
manifest=json.loads((root/'experiment.json').read_text());manifest['tests_passed']=57
manifest['changes']+=['canonical visual microbatches of8','PPO replays actually sampled discrete observation tokens with grounding STE gradients']
manifest['replay_diagnostic_before_fix']={'single_frame_max_ratio_error':2.384185791015625e-7,'batch_mean_ratio':1.1830671969801188,'batch_max_ratio_error':2.8940272331237793,'changed_bits':20}
(root/'experiment.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(source/'verification.log',root/'verification.log')
wrapper=root/'run_canonical_integration.py';wrapper.write_text((root/'run_integration.py').read_text().replace("root/'integration'","root/'integration_canonical'"))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFEREDINPUT='1',PYTHONUNBUFFERED='1')
with (root/'integration_canonical.log').open('ab') as log:
    p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(wrapper)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
result={'pid':p.pid,'time':time.time(),'out':str(root/'integration_canonical'),'gpu':6,'memory_fraction':.30}
(root/'integration_canonical_process.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
