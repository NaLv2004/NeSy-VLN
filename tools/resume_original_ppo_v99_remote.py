"""Resume the original PPO recipe from its preserved 43/80 checkpoint."""
import fcntl, hashlib, json, os, shutil, socket, subprocess, time
from pathlib import Path

B=Path('/home/user/xjy/nesy_vla_20261010')
old=B/'parallel_ppo_v85'; out=B/'original_ppo_resume_v99'
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out.mkdir(exist_ok=True)
lock=(out/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'launcher.json').exists(), 'Existing launch must be inspected, never duplicated'
stopped=read(B/'group_grpo_v98/USER_STOP.json')
assert stopped['state']=='stopped' and not stopped['remaining']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(out).free>8*2**30
source=old/'best_development'; commit=read(source/'COMMITTED'); best=read(old/'best_score.json')
assert commit['epoch']==70 and best['successes']==43
assert commit['checkpoint_sha256']=='449183eaca9abf0598549b1b1c70ad76ba9c80260252927215bbb03c9f6d3058'
assert sha(source/commit['checkpoint_file'])==commit['checkpoint_sha256']
shutil.copytree(old/'src',out/'src',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'),dirs_exist_ok=True)
snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
shutil.copy2(source/commit['checkpoint_file'],snapshot/commit['checkpoint_file'])
(snapshot/'COMMITTED').write_text(json.dumps(commit,indent=2))
shutil.copytree(source,out/'best_development',dirs_exist_ok=True)
(out/'best_score.json').write_text(json.dumps(dict(best,origin_run=str(old)),indent=2))
old_manifest=read(old/'manifest.json'); old_final=read(old/'latest/COMMITTED')
sample_start=old_manifest['sample_index_start']+(old_final['epoch']-old_manifest['parent_ppo_update'])*192
history={p.parent.name:read(p) for p in sorted((old/'evaluations').glob('step_*/summary.json'))}
record=dict(time=time.time(),checkpoint=str(snapshot),commit=commit,parent_run=str(old),
 development_before_parallel_change=best,source_development=best,sample_index_start=sample_start,
 source_original_run_final=old_final,old_validation_successes={k:v['successes'] for k,v in history.items()},
 reason='User requested continuation of the original training that produced 43/80. Restore original actor, critic and both optimizers; new training sampling RNG avoids replaying the already completed branch.',
 training_seed=9901,restore_optimizer=True,restore_critic=True,restore_rank_rng=False,
 maximum_additional_updates=100,development_interval=10,early_stop_no_improvement_validations=3,
 change_scope=['new output directory','new training sampling sequence','bounded continuation with early stopping'],
 no_change=['model architecture','PPO/GAE/loss weights','reward','train/development/fixed split','horizon32','revision10','eight GPUs and 24 trajectories per GPU'])
(out/'SOURCE.json').write_text(json.dumps(record,indent=2))
text=(out/'src/train.py').read_text()
assert text.count('parallel_ppo_v85')==4
text=text.replace('parallel_ppo_v85','original_ppo_resume_v99')
text=text.replace("p.add_argument('--updates',type=int,default=120)","p.add_argument('--updates',type=int,default=170)")
old_selection="Continue latest committed v82 eightGPU PPO model/optimizer/critic/allrankRNG. Only parallel rollout organization changes; no new baseline or fixed80 evaluation."
assert old_selection in text
text=text.replace(old_selection,'Continue the original PPO recipe from its development43/80 update70: actor, critic and both optimizers restored exactly; fresh training sampling RNG seed9901; no fresh baseline or fixed80 evaluation.')
anchor="rng.setstate(parent_rng['python']);torch.set_rng_state(parent_rng['torch']);torch.cuda.set_rng_state(parent_rng['cuda']);del parent_rng"
assert text.count(anchor)==1
text=text.replace(anchor,"del parent_rng\ncontinuation_seed=9901+1009*rank\nrng.seed(continuation_seed);torch.manual_seed(continuation_seed);torch.cuda.manual_seed(continuation_seed)")
anchor=" 'inverse_auxiliary':'Preserve parent observed-ID inverse supervision; no recurrent action-bearing belief inverse shortcut.',"
assert anchor in text
text=text.replace(anchor," 'training_sampling_seed':9901,'source_rank_rng_restored':False,'early_stop_no_improvement_validations':3,\n"+anchor)
anchor=" # The parent checkpoint was already evaluated when appropriate; do not replay it."
assert anchor in text
text=text.replace(anchor,anchor+"\n stale_validations=0;best_seen=read(out/'best_score.json')['success_rate'];last_completed=begin;completion_reason='update_limit'")
anchor="   if update%10==0:evaluate(update)\n progress(state='complete',update=args.updates)"
replacement="""   last_completed=update
   if update%10==0:
    summary=evaluate(update)
    if summary['success_rate']>best_seen:
     best_seen=summary['success_rate'];stale_validations=0
    else:stale_validations+=1
    if stale_validations>=3:
     completion_reason='three_validations_without_improvement'
     if rank==0:atomic(out/'EARLY_STOP.json',{'time':time.time(),'update':update,'best_success_rate':best_seen,'latest_success_rate':summary['success_rate'],'no_restart':True})
     break
 progress(state='complete',update=last_completed,completion_reason=completion_reason)"""
assert text.count(anchor)==1
text=text.replace(anchor,replacement)
compile(text,str(out/'src/train.py'),'exec')
(out/'src/train.py').write_text(text)
shutil.copy2(old/'supervise.py',out/'supervise.py')

verify=r'''
import sys,torch,json,hashlib
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/original_ppo_resume_v99')
sys.path.insert(0,str(b/'src'))
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.automaton_policy_gradient import SymbolicValue,configure_trainable
c=json.loads((b/'source_snapshot/COMMITTED').read_text())
s=torch.load(b/'source_snapshot'/c['checkpoint_file'],map_location='cpu',weights_only=False)
assert all(k in s for k in ['model','critic','optimizer','value_optimizer','rank_rng']) and len(s['rank_rng'])==8
m=ProbabilisticTokenAutomaton(**s['manifest']['architecture']);m.load_state_dict(s['model'])
critic=SymbolicValue(m.factors,m.states,m.tasks);critic.load_state_dict(s['critic'])
opt=torch.optim.AdamW(configure_trainable(m,'joint'),lr=3e-5,weight_decay=.0001);opt.load_state_dict(s['optimizer'])
value_opt=torch.optim.AdamW(critic.parameters(),lr=1e-3,weight_decay=.0001);value_opt.load_state_dict(s['value_optimizer'])
assert all(torch.equal(v,s['model'][k]) for k,v in m.state_dict().items())
assert all(torch.equal(v,s['critic'][k]) for k,v in critic.state_dict().items())
assert opt.param_groups[0]['lr']==3e-5 and value_opt.param_groups[0]['lr']==1e-3
from nesy_vla.runtime import DataBank,setup
cfg=json.loads(Path('/home/user/xjy/nesy_vla_20261010/symbolic_execution_v38/configs/grounded_outcome_feedback.json').read_text());setup(cfg);bank=DataBank(cfg)
assert bank.provenance==s['manifest']['data']
result={'source_update':s['epoch'],'source_sha256':c['checkpoint_sha256'],'actor_loaded_exactly':True,'critic_loaded_exactly':True,'actor_optimizer_states':len(opt.state),'critic_optimizer_states':len(value_opt.state),'actor_lr':opt.param_groups[0]['lr'],'critic_lr':value_opt.param_groups[0]['lr'],'split_provenance_unchanged':True,'train_tasks':len(bank.train),'development_tasks':len(bank.dev),'fixed_tasks':len(bank.fixed)}
(b/'RESTORE_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
result=subprocess.run([policy,'-c',verify],capture_output=True,text=True,timeout=60)
(out/'restore_verification.log').write_text(result.stdout+result.stderr)
assert result.returncode==0,result.stdout+result.stderr
prepared=dict(time=time.time(),trainer_sha256=sha(out/'src/train.py'),restore=read(out/'RESTORE_VERIFIED.json'),
 inherited_code_hashes={str(p.relative_to(out/'src')):sha(p) for p in (out/'src/nesy_vla').glob('*.py')})
(out/'PREPARED.json').write_text(json.dumps(prepared,indent=2))
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
launcher=dict(time=time.time(),world=8,jobs=[],algorithm='original_PPO_GAE',parent_update=70,
 maximum_final_update=170,global_rollouts_per_update=192,simultaneous_per_gpu=24,
 automatic_restart=False,source_checkpoint=commit,training_seed=9901,early_stop_validations=3)
for rank in range(8):
 folder=out/f'rank{rank}';folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),
  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',NCCL_DEBUG='WARN')
 with (folder/'supervisor.log').open('ab') as log:
  child=subprocess.Popen(['/usr/bin/python3','-u',str(out/'supervise.py')],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
 launcher['jobs'].append(dict(rank=rank,gpu=rank,supervisor_pid=child.pid))
 (out/'launcher.json').write_text(json.dumps(launcher,indent=2))
print(json.dumps(dict(source=record,restored=prepared['restore'],launcher=launcher),indent=2))
