import fcntl,json,os,signal,socket,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010/original_ppo_resume_v99');read=lambda p:json.loads(p.read_text())
lock=(B/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (B/'unlimited_launcher.json').exists(), 'Do not duplicate the unlimited trainer'
r=read(B/'UNLIMITED_REQUEST.json')
def remaining():
 out=[]
 for old in r['old_processes']:
  try:
   stat=Path(f"/proc/{old['pid']}/stat").read_text().rsplit(')',1)[1].split()
   if stat[19]==old['start'] and stat[0]!='Z':out.append(dict(pid=old['pid'],state=stat[0]))
  except FileNotFoundError:pass
 return out
deadline=time.monotonic()+40
while remaining() and time.monotonic()<deadline:time.sleep(.5)
if remaining():
 for p in remaining():
  try:os.kill(p['pid'],signal.SIGKILL)
  except ProcessLookupError:pass
 print(json.dumps({'state':'waiting_for_old_process_cleanup','remaining':remaining()}));raise SystemExit
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
old=read(B/'launcher.json');(B/'launcher_before_unlimited.json').write_text(json.dumps(old,indent=2))
launcher=dict(time=time.time(),world=8,jobs=[],algorithm='original_PPO_GAE',parent_update=70,
 maximum_final_update=None,global_rollouts_per_update=192,simultaneous_per_gpu=24,
 automatic_restart=False,source_checkpoint=old['source_checkpoint'],training_seed=9901,early_stop_validations=None,
 resume_commit=r['preserve_checkpoint'])
for rank in range(8):
 folder=B/f'rank{rank}'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),
  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',NCCL_DEBUG='WARN')
 with (folder/'supervisor.log').open('ab') as log:
  child=subprocess.Popen(['/usr/bin/python3','-u',str(B/'supervise.py')],cwd=B,env=env,stdout=log,stderr=log,start_new_session=True)
 launcher['jobs'].append(dict(rank=rank,gpu=rank,supervisor_pid=child.pid))
 (B/'launcher.json').write_text(json.dumps(launcher,indent=2))
(B/'unlimited_launcher.json').write_text(json.dumps(launcher,indent=2))
r.update(status='unlimited_training_launched',launched_at=time.time())
(B/'UNLIMITED_REQUEST.json').write_text(json.dumps(r,indent=2))
print(json.dumps(launcher,indent=2))
