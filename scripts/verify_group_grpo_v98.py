"""Two-rank tensor-only GRPO invariants on archived training observations."""
import copy,datetime,hashlib,json,os,random,sys
from pathlib import Path
import torch
import torch.distributed as dist
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'group_grpo_v98';sys.path.insert(0,str(out/'src'))
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.automaton_policy_gradient import replay
from nesy_vla.group_relative import ActorSnapshot,group_advantages,group_objective
from nesy_vla.distributed_symbolic import average_gradients,assert_identical_parameters,gather_objects
from nesy_vla.symbolic_trust_region import policy_drift
rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
torch.set_num_threads(2);torch.manual_seed(9800+rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
read=lambda p:json.loads(p.read_text())
source=out/'source_snapshot';commit=read(source/'COMMITTED');path=source/commit['checkpoint_file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==commit['checkpoint_sha256']
saved=torch.load(path,map_location='cpu',weights_only=False)
model=ProbabilisticTokenAutomaton(**saved['manifest']['architecture']).cuda();model.load_state_dict(saved['model'])
reference=copy.deepcopy(model).eval().requires_grad_(False)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.0001)
trial=read(sorted((B/f'parallel_ppo_v85/rollouts/step_000061/rank{rank}').glob('*.json'))[0])
assert trial['episode']['scene'] in saved['manifest']['data']['train_layouts']
length=min(12,len(trial['steps']))
x=torch.tensor([[s['patch_tokens'] for s in trial['steps'][:length]]]*8,device='cuda',dtype=torch.long)
task=torch.full((8,),['chair','table'].index(trial['episode']['category']),device='cuda',dtype=torch.long)
a=torch.zeros(8,length,device='cuda',dtype=torch.long);valid=torch.ones_like(a,dtype=torch.bool)
with torch.no_grad():
 for step in range(length):
  r=replay(model,x[:,:step+1],task,a[:,:step+1])
  a[:,step]=torch.multinomial(r['probabilities'][:,-1],1).squeeze(-1)
 old=replay(model,x,task,a)['probabilities'];ref=replay(reference,x,task,a)['probabilities']
 assert torch.equal(old,ref)
 old_logp=old.clamp_min(1e-20).log().gather(-1,a[...,None]).squeeze(-1)
 # Synthetic ranking tests tensor plumbing only, never saved as training data.
 rewards=torch.zeros_like(a,dtype=torch.float);rewards[:,-1]=torch.arange(8.,device='cuda')
 adv,group=group_advantages(rewards,valid)
rng=random.Random(98+rank);snapshot=ActorSnapshot(model,optimizer,rng)
r=replay(model,x,task,a);loss,audit=group_objective(r['probabilities'],a,old_logp,adv,valid,ref)
assert audit['max_ratio_error']==0
loss.backward();average_gradients(model.parameters())
gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert 0<float(gradient)<float('inf')
optimizer.step();assert_identical_parameters(model)
with torch.no_grad():
 updated=replay(model,x,task,a)['probabilities'];stats=policy_drift(old,updated,a,valid);dist.all_reduce(stats)
 assert float((updated-old).abs().max())>0
kl=float(stats[0]/stats[2]);assert 0<kl<.02,kl
snapshot.restore();assert torch.equal(replay(model,x,task,a)['probabilities'],old)
assert not optimizer.state;assert_identical_parameters(model)
assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.state_dict().items())
proof_error=0.
with torch.no_grad():
 _,memory=model.initial_state(8,'cuda');last=torch.full((8,),17,device='cuda',dtype=torch.long)
 for step in range(length):
  if step%10==0:program=model.generate(task,x[:,max(0,step-10):step+1],step)
  result=model.step(program,x[:,step],memory,last,step)
  proof=model.proof(program,result['observed_ids'],memory,last,result['state_values'],step)
  proof_error=max(proof_error,float((proof['probabilities']-result['probabilities']).abs().max()))
  memory=result['state_values'];last=a[:,step]
assert proof_error==0
results=gather_objects(dict(rank=rank,source_training_episode=trial['episode']['episode_id'],actions=int(valid.sum()),
 group_return_std=group['return_std'],gradient_norm=float(gradient),old_probability_error=audit['max_ratio_error'],
 update_kl=kl,rollback_exact=True,reference_unchanged=True,symbolic_proof_error=proof_error,parameter_sync_exact=True))
if rank==0:
 value=dict(source=commit,ranks=results,scope='Tensor-only invariants. Synthetic rewards never used for real training. No new navigation evaluation.',heldout_data_used=False)
 (out/'integration.json').write_text(json.dumps(value,indent=2));print(json.dumps(value,indent=2))
dist.destroy_process_group()
