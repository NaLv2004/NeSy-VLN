import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';out=root/'integration'
out.mkdir(exist_ok=True);marker=out/'process.json'
if marker.exists():
    old=json.loads(marker.read_text());p=Path(f"/proc/{old['pid']}/cmdline")
    if p.exists() and b'integration.py' in p.read_bytes():print(json.dumps(old));raise SystemExit
if (out/'latest/COMMITTED').exists():raise RuntimeError('Already committed integration; inspect instead of rerunning')
cfg=json.loads((base/'symbolic_reward_v11/configs/original.json').read_text())
cfg.update(root=str(root),updates=1,rollouts_per_rank=1,demonstrations_per_rank=4,
    paired_demonstration_roots=[str(base/'paired_goal_probe_v12/input_snapshot')],paired_demo_fraction=.5,
    training_chunk_batch=2,maximum_chunk_batch=2,training_frame_budget=20,revision_chunk_limit=2,
    candidate_every=1,architecture='paired_goal_v12_integration')
(root/'configs').mkdir(exist_ok=True);(root/'configs/integration.json').write_text(json.dumps(cfg,indent=2))
wrapper=root/'integration.py';wrapper.write_text("import runpy,sys,torch\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root))\ntorch.cuda.set_per_process_memory_fraction(.18)\nsys.argv=['scripts/train.py','--config','configs/integration.json','--out',str(root/'integration')]\nrunpy.run_path(str(root/'scripts/train.py'),run_name='__main__')\n")
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(wrapper)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (out/'integration.log').open('ab') as log:
    child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':child.pid,'time':time.time(),'command':cmd};marker.write_text(json.dumps(record,indent=2));print(json.dumps(record))
