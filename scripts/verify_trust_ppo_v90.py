"""Two-rank optimizer integration using immutable recorded training only."""
import datetime,hashlib,json,os,random,sys
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'trust_ppo_v90';sys.path.insert(0,str(out/'src'))
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.automaton_policy_gradient import SymbolicValue,replay,advantages,policy_objective
from nesy_vla.distributed_symbolic import average_gradients,assert_identical_parameters,global_moments,gather_objects
from nesy_vla.symbolic_trust_region import UpdateSnapshot,policy_drift
rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
torch.set_default_device('cuda');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
read=lambda p:json.loads(p.read_text())
source=B/'parallel_ppo_v85/step_000060';commit=read(source/'COMMITTED');path=source/commit['checkpoint_file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==commit['checkpoint_sha256']
saved=torch.load(path,map_location='cpu',weights_only=False)
model=ProbabilisticTokenAutomaton(**saved['manifest']['architecture']).cuda();model.load_state_dict(saved['model'])
critic=SymbolicValue(model.factors,model.states,model.tasks).cuda();critic.load_state_dict(saved['critic'])
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.load_state_dict(saved['optimizer'])
value_optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-3);value_optimizer.load_state_dict(saved['value_optimizer'])
for g in optimizer.param_groups:g['lr']=1e-4
trials=[read(p) for p in sorted((B/f'parallel_ppo_v85/rollouts/step_000061/rank{rank}').glob('*.json'))[:8]]
length=max(len(t['steps']) for t in trials);x=torch.zeros(8,length,49,dtype=torch.long);a=torch.zeros(8,length,dtype=torch.long)
task=torch.tensor([['chair','table'].index(t['episode']['category']) for t in trials]);valid=torch.zeros(8,length,dtype=torch.bool);rewards=torch.zeros(8,length)
for i,t in enumerate(trials):
 n=len(t['steps']);valid[i,:n]=True;x[i]=torch.tensor([s['patch_tokens'] for s in t['steps']]+[t['steps'][-1]['patch_tokens']]*(length-n))
 a[i,:n]=torch.tensor([s['action'] for s in t['steps']]);rewards[i,:n]=torch.tensor([s['reward'] for s in t['steps']])
with torch.no_grad():
 original=replay(model,x,task,a);old=original['probabilities'];goals=task[:,None].expand_as(a)
 values=critic(original['beliefs'].flatten(0,1),goals.flatten()).reshape_as(rewards)
 adv,returns=advantages(rewards,values,valid)
 error=max(float((old[i,:len(t['steps'])]-torch.tensor([s['probabilities'] for s in t['steps']])).abs().max()) for i,t in enumerate(trials))
 assert error==0,error
mean,std=global_moments(adv[valid]);global_actions=torch.tensor(int(valid.sum()));dist.all_reduce(global_actions)
rank_weight=dist.get_world_size()*int(valid.sum())/int(global_actions)
rng=random.Random(9000+rank);snapshot=UpdateSnapshot(model,optimizer,critic,value_optimizer,rng)
optimizer.zero_grad(set_to_none=True);value_optimizer.zero_grad(set_to_none=True)
r=replay(model,x,task,a);oldlog=old.clamp_min(1e-20).log().gather(-1,a[...,None]).squeeze(-1)
loss,_=policy_objective(r['probabilities'],a,oldlog,(adv-mean)/std,valid)
(loss*rank_weight).backward();average_gradients(model.parameters());torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
v=critic(r['beliefs'].flatten(0,1),goals.flatten()).reshape_as(returns)
(rank_weight*F.smooth_l1_loss(v[valid],returns[valid])).backward();average_gradients(critic.parameters());torch.nn.utils.clip_grad_norm_(critic.parameters(),5.)
optimizer.step();value_optimizer.step()
with torch.no_grad():
 updated=replay(model,x,task,a);stats=policy_drift(old,updated['probabilities'],a,valid);dist.all_reduce(stats)
mean_kl=float(stats[0]/stats[2]);assert 0<mean_kl<.02,mean_kl
assert_identical_parameters(model);assert_identical_parameters(critic)
snapshot.restore()
assert torch.equal(replay(model,x,task,a)['probabilities'],old)
assert_identical_parameters(model);assert_identical_parameters(critic)
with torch.no_grad():
 model.conclusions[...,0].add_(50.)
 bad=replay(model,x,task,a);bad_stats=policy_drift(old,bad['probabilities'],a,valid);dist.all_reduce(bad_stats)
bad_kl=float(bad_stats[0]/bad_stats[2]);assert bad_kl>.02
snapshot.restore()
assert torch.equal(replay(model,x,task,a)['probabilities'],old)
assert_identical_parameters(model);assert_identical_parameters(critic)
proof_error=0.
with torch.no_grad():
 _,memory=model.initial_state(8,'cuda');last=torch.full((8,),17,dtype=torch.long)
 for step in range(length):
  if step%10==0:program=model.generate(task,x[:,max(0,step-10):step+1],step)
  result=model.step(program,x[:,step],memory,last,step)
  proof=model.proof(program,result['observed_ids'],memory,last,result['state_values'],step)
  proof_error=max(proof_error,float((proof['probabilities']-result['probabilities']).abs().max()))
  memory=result['state_values'];last=a[:,step]
assert proof_error==0
rows=gather_objects(dict(rank=rank,training_episodes=8,actions=int(valid.sum()),horizon=length,
 old_probability_error=error,accepted_candidate_kl=mean_kl,rejected_candidate_kl=bad_kl,
 rollback_exact=True,symbolic_proof_error=proof_error,actor_and_critic_synchronization_exact=True))
if rank==0:
 result=dict(source=commit,ranks=rows,fresh_navigation_rollouts=0,heldout_data_used=False)
 (out/'integration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
dist.destroy_process_group()
