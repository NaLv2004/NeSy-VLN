"""Start one eight-rank GRPO experiment after the PPO processes are gone."""
import fcntl,hashlib,json,os,shutil,socket,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'group_grpo_v98';old=B/'trust_ppo_v90'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=(out/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'launcher.json').exists(),'Inspect existing handles; do not launch twice'
prepared=read(out/'PREPARED.json');assert sha(out/'src/train.py')==prepared['trainer_sha256']
for name,value in prepared['source_hashes'].items():assert sha(out/'src'/name)==value,name
for j in read(old/'launcher.json')['jobs']:
 folder=old/f"rank{j['rank']}";s=read(folder/'supervisor.json')
 assert s['state']=='complete'
 for pid in [j['supervisor_pid'],s['pid']]+read(folder/'processes.json')['simulator_pids']:
  assert not Path(f'/proc/{pid}/cmdline').exists(),('Old PPO process still exists',pid)
source=read(out/'SOURCE.json');commit=read(out/'source_snapshot/COMMITTED')
assert commit==source['commit'] and sha(out/'source_snapshot'/commit['checkpoint_file'])==commit['checkpoint_sha256']
assert source['selected_development']['successes']==43 and commit['epoch']==70
assert shutil.disk_usage(out).free>8*2**30
available=[int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
assert available>75*1024**2,available
gpu_free={}
for gpu in range(8):
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 assert free>20000,(gpu,free)
 gpu_free[gpu]=free
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
launcher=dict(time=time.time(),world=8,jobs=[],updates=120,algorithm='GRPO',group_size=8,
 global_groups_per_update=24,global_rollouts_per_update=192,simultaneous_rollouts_per_gpu=24,
 source_checkpoint=commit,source_development_successes=43,experiment_count=1,automatic_restart=False,
 gpu_free_before_launch=gpu_free,maximum_actions=32)
for rank in range(8):
 folder=out/f'rank{rank}';folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),
  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',NCCL_DEBUG='WARN')
 with (folder/'supervisor.log').open('ab') as log:
  child=subprocess.Popen(['/usr/bin/python3','-u',str(out/'supervise.py')],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
 launcher['jobs'].append(dict(rank=rank,gpu=rank,supervisor_pid=child.pid))
 (out/'launcher.json').write_text(json.dumps(launcher,indent=2))
(out/'HANDOFF.json').write_text(json.dumps(dict(source=source,launcher=launcher),indent=2))
(old/'USER_SWITCHED_TO_GRPO.json').write_text(json.dumps({'time':time.time(),'successor':str(out),'old_ppo_completed':True,'do_not_restart':True},indent=2))
print(json.dumps(launcher,indent=2))
