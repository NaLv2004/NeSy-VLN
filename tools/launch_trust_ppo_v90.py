"""Start one successor only after the previous eight-GPU run has completed."""
import fcntl,hashlib,json,os,shutil,socket,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');old=B/'parallel_ppo_v85';out=B/'trust_ppo_v90'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
lock=(out/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'launcher.json').exists(),'Inspect the existing successor handles; never duplicate launch'
prepared=read(out/'PREPARED.json');assert sha(out/'src/train.py')==prepared['trainer_sha256']
for name,value in prepared['source_hashes'].items():assert sha(out/'src'/name)==value
old_jobs=read(old/'launcher.json')['jobs'];evidence=[];active=[]
for j in old_jobs:
 folder=old/f'rank{j["rank"]}';supervisor=read(folder/'supervisor.json');status=read(folder/'status.json')
 ids=[j['supervisor_pid'],supervisor['pid']]+read(folder/'processes.json')['simulator_pids']
 alive={pid:cmd(pid) for pid in ids if cmd(pid)}
 evidence.append(dict(rank=j['rank'],supervisor=supervisor,status_state=status['state'],update=status.get('update'),alive=alive))
 if alive or supervisor['state']!='complete' or supervisor.get('exit_code')!=0 or status['state']!='complete':active.append(evidence[-1])
if active:
 print(json.dumps({'waiting_for_previous_run':active,'time':time.time()},indent=2));raise SystemExit
terminal=read(old/'latest/COMMITTED');assert terminal['epoch']==120
assert sha(old/'latest'/terminal['checkpoint_file'])==terminal['checkpoint_sha256']
best=read(old/'best_score.json');selected=old/'best_development';commit=read(selected/'COMMITTED')
assert commit['epoch']==best['epoch'] and best['episodes']==80
assert sha(selected/commit['checkpoint_file'])==commit['checkpoint_sha256']
assert shutil.disk_usage(out).free>8*2**30
available=[int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
assert available>75*1024**2,available
for gpu in range(8):
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 assert free>20000,(gpu,free)
snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
shutil.copy2(selected/commit['checkpoint_file'],snapshot/commit['checkpoint_file'])
(snapshot/'COMMITTED').write_text(json.dumps(commit,indent=2));assert sha(snapshot/commit['checkpoint_file'])==commit['checkpoint_sha256']
parent=read(old/'manifest.json')
record=dict(time=time.time(),checkpoint=str(snapshot),commit=commit,parent_run=str(old),parent_terminal_commit=terminal,
 selected_development=best,development_before_parallel_change=best,
 sample_index_start=parent['sample_index_start']+(commit['epoch']-parent['parent_ppo_update'])*192,
 selection='Best complete development success; existing earliest-best tie rule. No originalfixed80 inspected or rerun.',
 training_change='Initial actorLR1e-4, atmost6optimizerpasses, exact17-action meanKLtarget.01/limit.02; rollback andhalveLRonovershoot. Other objectives/data/32horizon/revise10/symbolic architecture unchanged.',
 source_optimizer_and_all_rank_rng_retained=True,new_stage_counter_starts_at_zero=True,old_processes=evidence)
(out/'SOURCE.json').write_text(json.dumps(record,indent=2))
shutil.copytree(selected,out/'best_development',dirs_exist_ok=True)
best=dict(best,origin_run=best.get('origin_run',str(old)))
(out/'best_score.json').write_text(json.dumps(best,indent=2))
(out/'HANDOFF.json').write_text(json.dumps(record,indent=2))
(out/'supervise.py').write_text((old/'supervise.py').read_text())
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
launcher=dict(time=time.time(),world=8,jobs=[],updates=120,parent_selected_update=commit['epoch'],
 global_rollouts_per_update=192,simultaneous_rollouts_per_gpu=24,experiment_count=1,automatic_restart=False)
for rank in range(8):
 folder=out/f'rank{rank}';folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),
  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',NCCL_DEBUG='WARN')
 with (folder/'supervisor.log').open('ab') as log:
  child=subprocess.Popen(['/usr/bin/python3','-u',str(out/'supervise.py')],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
 launcher['jobs'].append(dict(rank=rank,gpu=rank,supervisor_pid=child.pid))
 (out/'launcher.json').write_text(json.dumps(launcher,indent=2))
print(json.dumps({'selected_checkpoint':commit,'development':best,'launcher':launcher},indent=2))
