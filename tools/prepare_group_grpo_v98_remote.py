import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'group_grpo_v98';stage=out/'staging';src=out/'src'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'launcher.json').exists(),'Never overwrite a launched experiment'
(out/'PREPARATION_PROCESS.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time()}))
old=B/'trust_ppo_v90';terminal=read(old/'latest/COMMITTED')
assert terminal['epoch']==120 and sha(old/'latest'/terminal['checkpoint_file'])==terminal['checkpoint_sha256']
old_jobs=[]
for job in read(old/'launcher.json')['jobs']:
 folder=old/f"rank{job['rank']}";s=read(folder/'supervisor.json');state=read(folder/'status.json')
 assert s['state']=='complete' and s['exit_code']==0 and state['update']==120
 ids=[job['supervisor_pid'],s['pid']]+read(folder/'processes.json')['simulator_pids']
 live={pid:Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace') for pid in ids if Path(f'/proc/{pid}/cmdline').exists()}
 assert not live,live
 old_jobs.append(dict(rank=job['rank'],state='complete',update=120,live_processes=live))
source=old/'source_snapshot';commit=read(source/'COMMITTED')
assert commit['epoch']==70 and commit['checkpoint_sha256']=='449183eaca9abf0598549b1b1c70ad76ba9c80260252927215bbb03c9f6d3058'
assert sha(source/commit['checkpoint_file'])==commit['checkpoint_sha256']
parent=read(old/'SOURCE.json');assert parent['selected_development']['successes']==43
snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
shutil.copy2(source/commit['checkpoint_file'],snapshot/commit['checkpoint_file'])
(snapshot/'COMMITTED').write_text(json.dumps(commit,indent=2))
record=dict(time=time.time(),checkpoint=str(snapshot),commit=commit,parent_run=str(B/'parallel_ppo_v85'),
 selected_development=parent['selected_development'],sample_index_start=parent['sample_index_start'],
 previous_ppo_terminal_commit=terminal,previous_ppo_terminal_successes=read(old/'evaluations/step_000120/summary.json')['successes'],
 previous_jobs=old_jobs,selection='User requested GRPO from preserved PPO70 development43/80. Fresh actor optimizer, no critic.',
 group_size=8,global_groups=24,global_rollouts=192,horizon=32,development_interval=10,fixed80_used=False)
(out/'SOURCE.json').write_text(json.dumps(record,indent=2))
shutil.copytree(old/'src',src,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'),dirs_exist_ok=True)
for name,target in [('train_group_grpo_v98.py','train.py'),('group_relative.py','nesy_vla/group_relative.py'),
 ('grouped_rollout.py','nesy_vla/grouped_rollout.py'),('test_group_relative.py','test_group_relative.py')]:shutil.copy2(stage/name,src/target)
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
test=subprocess.run([policy,'-m','pytest','test_group_relative.py','test_automaton_policy_gradient.py','test_symbolic_trust_region.py','-q'],cwd=src,env=env,capture_output=True,text=True)
(out/'test_results.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
env['CUDA_VISIBLE_DEVICES']='0,1'
result=subprocess.run([policy,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(stage/'verify_group_grpo_v98.py')],cwd=src,env=env,capture_output=True,text=True)
(out/'integration.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stdout+result.stderr
shutil.copytree(snapshot,out/'best_development',dirs_exist_ok=True)
(out/'best_score.json').write_text(json.dumps(dict(parent['selected_development'],origin_run=str(B/'parallel_ppo_v85')),indent=2))
shutil.copy2(old/'supervise.py',out/'supervise.py')
record=dict(time=time.time(),tests=test.stdout,integration=read(out/'integration.json'),
 trainer_sha256=sha(src/'train.py'),source_hashes={p.relative_to(src).as_posix():sha(p) for p in (src/'nesy_vla').glob('*.py')})
(out/'PREPARED.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
