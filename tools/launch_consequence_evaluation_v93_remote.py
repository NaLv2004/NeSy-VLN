"""Evaluate one refinement checkpoint once, only after its trainer is terminal."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
B=Path('/home/user/xjy/nesy_vla_20261010');OUT=B/'consequence_refinement_v93'
sys.path.insert(0,str(OUT/'src'))
from nesy_vla.small_checkpoint import save_small_checkpoint
read=lambda p:json.loads(p.read_text())
lock=(OUT/'evaluation_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'evaluation_process.json').exists(),'Inspect existing evaluator; do not launch it twice'
process=read(OUT/'process.json')
assert not Path(f'/proc/{process["pid"]}/cmdline').exists(),'Refinement trainer is still live'
done=read(OUT/'COMPLETE');assert done['completed_updates']==400
assert read(OUT/'status.json')['state']=='complete'
selected=read(OUT/'best_component.json');update=selected['update']
source=OUT/f'checkpoint_{update:04d}.pt'
payload=torch.load(source,map_location='cpu',weights_only=False)
assert payload['update']==update and payload['manifest']['probe_layouts']==['layout_118','layout_119']
assert payload['manifest']['architecture']['states']==32
manifest=dict(payload['manifest'],source_commit=payload['manifest']['source_tokens'])
folder=OUT/'evaluation_checkpoint'
if not (folder/'COMMITTED').exists():
 commit=save_small_checkpoint(folder,dict(model=payload['model'],epoch=update,manifest=manifest),
  dict(epoch=update,horizon=32,revision_interval=10,source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
  reservation_root=OUT,wait_seconds=0)
else:commit=read(folder/'COMMITTED')
del payload
evaluator=OUT/'evaluate.py';assert evaluator.exists()
# Eight simulators need CPU as well as GPU memory; don't launch if the host
# cannot accommodate them alongside the unchanged main training run.
mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
assert mem['MemAvailable']>6*1024**2,('Insufficient available host RAM',mem['MemAvailable'])
free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
assert free>5000,('Insufficient GPU1 memory',free)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1')
with (OUT/'evaluation.log').open('ab') as log:
 child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(evaluator),
  '--checkpoint',str(folder),'--out',str(OUT/'development'),'--suite','development'],
  cwd=OUT,env=env,stdout=log,stderr=log,start_new_session=True)
record=dict(time=time.time(),pid=child.pid,gpu=1,checkpoint=commit,selected_component=selected,
 original_fixed80_used=False,main_ppo_stopped=False,selection='Only training-layout component scores; no development comparison among candidates',
 evaluator_sha256=hashlib.sha256(evaluator.read_bytes()).hexdigest())
(OUT/'evaluation_process.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
