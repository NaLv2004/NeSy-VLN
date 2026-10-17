"""Snapshot committed PPO state and replace only its eight process groups."""
import hashlib,json,os,shutil,signal,socket,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');old=B/'symbolic_ppo8_v82';out=B/'parallel_ppo_v84'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (out/'PREPARED.json').exists() and not (out/'launcher.json').exists()
def cmd(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
jobs=read(old/'launcher.json')['jobs'];groups=[];all_pids=[]
for job in jobs:
 status=read(old/f"rank{job['rank']}"/'supervisor.json');supervisor=job['supervisor_pid'];trainer=status['pid']
 assert str(old/'supervise.py') in cmd(supervisor) and os.getpgid(supervisor)==supervisor
 assert str(old/'src/train.py') in cmd(trainer) and os.getpgid(trainer)==supervisor
 groups.append(supervisor);all_pids.extend([supervisor,trainer])
 all_pids.extend(read(old/f"rank{job['rank']}"/'processes.json')['simulator_pids'])
try:
 for pgid in groups:os.killpg(pgid,signal.SIGSTOP)
 commit=read(old/'latest/COMMITTED');assert sha(old/'latest'/commit['checkpoint_file'])==commit['checkpoint_sha256']
 assert 0<commit['epoch']<120
 source=out/'source_snapshot';source.mkdir(exist_ok=True)
 shutil.copy2(old/'latest'/commit['checkpoint_file'],source/commit['checkpoint_file'])
 (source/'COMMITTED').write_text(json.dumps(commit,indent=2))
 completed=sorted(old.glob('evaluations/step_*/summary.json'));assert completed
 record={'time':time.time(),'checkpoint':str(source),'commit':commit,'parent_run':str(old),
  'development_before_parallel_change':read(completed[-1]),'last_development_step':completed[-1].parent.name,
  'parallel_before':8,'parallel_after':24,'source_training_rollouts_per_global_update':128,
  'reason':'User requested higher simultaneous rollout concurrency and continued PPO; same actor, optimizer, critic, reward and isolated splits.'}
 (out/'SOURCE.json').write_text(json.dumps(record,indent=2))
 # Retain previous selected best as well as the exact continuation state.
 shutil.copytree(old/'best_development',out/'best_development',dirs_exist_ok=True)
 prior_best=read(old/'best_score.json');prior_best['origin_run']=str(old)
 (out/'best_score.json').write_text(json.dumps(prior_best,indent=2))
 (old/'PARALLEL_HANDOFF.json').write_text(json.dumps(record,indent=2))
except BaseException:
 for pgid in groups:
  try:os.killpg(pgid,signal.SIGCONT)
  except ProcessLookupError:pass
 raise
for pgid in groups:
 os.killpg(pgid,signal.SIGTERM);os.killpg(pgid,signal.SIGCONT)
deadline=time.time()+15
while any(cmd(pid) for pid in all_pids) and time.time()<deadline:time.sleep(.2)
assert not any(cmd(pid) for pid in all_pids),[(pid,cmd(pid)) for pid in all_pids if cmd(pid)]
record['all_old_process_handles_absent']=True;record['old_process_ids']=all_pids
(out/'HANDOFF.json').write_text(json.dumps(record,indent=2))
assert shutil.disk_usage(out).free>8*2**30
supervisor=(old/'supervise.py').read_text();(out/'supervise.py').write_text(supervisor)
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
result={'time':time.time(),'world':8,'jobs':[],'updates':120,'parent_update':commit['epoch'],
 'global_rollouts_per_update':192,'simultaneous_rollouts_per_gpu':24,'experiment_count':1,'automatic_restart':False}
for rank in range(8):
 folder=out/f'rank{rank}';folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),
  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',NCCL_DEBUG='WARN')
 with (folder/'supervisor.log').open('ab') as log:
  child=subprocess.Popen(['/usr/bin/python3','-u',str(out/'supervise.py')],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
 result['jobs'].append({'rank':rank,'gpu':rank,'supervisor_pid':child.pid})
 (out/'launcher.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'handoff_update':commit['epoch'],'checkpoint_sha256':commit['checkpoint_sha256'],'launcher':result},indent=2))
