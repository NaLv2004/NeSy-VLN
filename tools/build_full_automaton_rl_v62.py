"""Build a frozen experiment using the already audited data/evaluation path."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'scripts/train_probabilistic_automaton_v56.py').read_text()
prefix=s[:s.index('p=argparse.ArgumentParser()')]
prefix+='from nesy_vla.automaton_policy_gradient import SymbolicValue,replay,advantages,policy_objective,configure_trainable\nfrom nesy_vla.reward_experiment import training_reward\n'
data=s[s.index('torch.set_num_threads(2)'):s.index('architecture={')]
bc=s[s.index('all_codes='):s.index('first_rows={')]
bc=bc.replace('def train_batch(batch):','def demonstration_loss(batch):')
bc=bc[:bc.index('    optimizer.zero_grad(set_to_none=True)')]+'    return total\n'
evaluation=s[s.index('vision=None;'):s.index('\ntry:\n    if begin')]
evaluation=evaluation.replace("images=[pil(r['rgb']) for r in responses];task=torch.tensor([tasks.index(e['category']) for e in batch],device='cuda')", "images=[pil(r['rgb']) for r in responses];images+=images[:1]*(8-n);task=torch.tensor([tasks.index(e['category']) for e in batch+[batch[0]]*(8-n)],device='cuda')")
evaluation=evaluation.replace("model.initial_state(n,'cuda');last_action=torch.full((n,),", "model.initial_state(8,'cuda');last_action=torch.full((8,),")
initial=r'''
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['joint','action_rules_only'],required=True)
p.add_argument('--updates',type=int,default=80);args=p.parse_args()
out=base/'full_automaton_rl_v62'/args.arm;out.mkdir(parents=True,exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'status.json').exists() or read(out/'status.json')['state']!='complete'
'''
initial+=data
initial+=r'''
warm_source=base/'probabilistic_automaton_v56/soft_recurrent/step_000040'
warm=load_committed(warm_source);architecture=warm['manifest']['architecture']
assert not architecture['reset_memory'] and warm['manifest']['data']==bank.provenance
model=ProbabilisticTokenAutomaton(**architecture).cuda();model.load_state_dict(warm['model']);del warm
trainable=configure_trainable(model,args.arm)
critic=SymbolicValue(model.factors,model.states,model.tasks).cuda()
optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=.0001)
value_optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-3,weight_decay=.0001)
manifest={'architecture':architecture,'arm':args.arm,'warm_start_checkpoint':str(warm_source),
 'warm_start_commit':read(warm_source/'COMMITTED'),'data':bank.provenance,'horizon':32,'revision_interval':10,
 'updates':args.updates,'episodes_per_update':16,'canonical_symbolic_batch':8,
 'actor_lr':1e-4,'critic_lr':1e-3,'ppo_epochs':2,'clip':.2,'gamma':.99,'gae_lambda':.95,
 'entropy_weight':.01,'demonstration_anchor':.1,'auxiliary_critic_gradient_to_actor':False,
 'train_rows_sha256':sha(source/'training_rows.json'),'recovery_commit':extra_commit,
 'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
 'reward_overrides':cfg.get('training_reward_overrides'),'fixed80_used':False,
 'source_development':read(warm_source.parent/'evaluations/step_000040/summary.json'),
 'action_path':'Integer visual vocabulary -> learned integer observation IDs -> explicit probabilistic finite-state transitions -> pairwise equality rules -> action distribution. Program revisions every10 actions use integer token history only.',
 'comparison':'Identical source, task sampler and sampling seed. Joint trains all learned states/rules/generator; control trains only action-conclusion tables and rule strengths.',
 'optimizer_resume':False,'source_sft_optimizer_retained':True}
if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
else:atomic(out/'manifest.json',manifest)
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
begin=0
if (out/'latest/COMMITTED').exists():
 saved=load_committed(out/'latest');assert saved['manifest']==manifest
 model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
 critic.load_state_dict(saved['critic']);value_optimizer.load_state_dict(saved['value_optimizer']);begin=saved['epoch']
 rng.setstate(saved['rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])

def save(folder,epoch):
 payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'critic':critic.state_dict(),
  'value_optimizer':value_optimizer.state_dict(),'epoch':epoch,'manifest':manifest,'rng':rng.getstate(),
  'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
 committed_torch_save(folder,payload,{'epoch':epoch,'horizon':32,'revision_interval':10,'optimizer_included':True})
'''
rl=r'''
@torch.no_grad()
def collect(batch,pool):
 assert len(batch)==8
 model.train()
 responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(sims,batch)))
 images=[pil(r['rgb']) for r in responses];task=torch.tensor([tasks.index(e['category']) for e in batch],device='cuda')
 _,memory=model.initial_state(8,'cuda');previous_action=torch.full((8,),17,dtype=torch.long,device='cuda')
 active=list(range(8));history=[];chosen=[];valid=[];probabilities=[];values=[];rewards=[];proof_error=0.
 trials=[{'episode':e,'steps':[],'programs':[]} for e in batch]
 for step in range(32):
  x=torch.stack([encode(image) for image in images]);history.append(x)
  if step%10==0:
   program=model.generate(task,torch.stack(history[-11:],1),step)
   for i in active:trials[i]['programs'].append({'start':step,'coefficients':program.coefficients[i].tolist()})
  result=model.step(program,x,memory,previous_action,step)
  proof=model.proof(program,result['observed_ids'],memory,previous_action,result['state_values'],step)
  proof_error=max(proof_error,float((proof['probabilities']-result['probabilities']).abs().max()));assert proof_error<1e-7
  action=torch.multinomial(result['probabilities'],1).squeeze(1)
  flag=torch.zeros(8,dtype=torch.bool,device='cuda');flag[active]=True;valid.append(flag)
  chosen.append(action);probabilities.append(result['probabilities']);values.append(critic(result['state_values'],task))
  for i in active:
   trials[i]['steps'].append({'step':step,'patch_tokens':x[i].tolist(),'observed_ids':result['observed_ids'][i].tolist(),
    'state_ids':result['state_ids'][i].tolist(),'action':int(action[i]),'probabilities':result['probabilities'][i].tolist(),
    'program_start':program.start})
  memory=result['state_values'];previous_action=action
  responses=list(pool.map(lambda pair:sims[pair[0]].call('step',action=pair[1]),[(i,int(action[i])) for i in active]))
  reward=torch.zeros(8,device='cuda');remaining=[]
  for i,response in zip(active,responses):
   adjusted=training_reward(response,cfg.get('training_reward_overrides'));reward[i]=float(adjusted['reward'])
   trials[i]['steps'][-1].update(reward=float(reward[i]),collision=bool(response.get('feedback',{}).get('collision',False)),
    reward_terms=adjusted.get('training_reward_audit',{}).get('training_terms',response.get('metrics',{}).get('reward_terms',{})))
   trials[i]['metrics']=response.get('metrics',{});images[i]=pil(response['rgb'])
   if not response['done']:remaining.append(i)
  rewards.append(reward);active=remaining
  if not active:break
 x=torch.stack(history,1);a=torch.stack(chosen,1);v=torch.stack(values,1);mask=torch.stack(valid,1)
 old=torch.stack(probabilities,1).clamp_min(1e-20).log().gather(-1,a[...,None]).squeeze(-1)
 adv,returns=advantages(torch.stack(rewards,1),v,mask)
 # This check is before any optimizer update. It detects batch/layout/mode drift.
 reproduced=replay(model,x,task,a);now=reproduced['probabilities'].clamp_min(1e-20).log().gather(-1,a[...,None]).squeeze(-1)
 error=float(((now-old).exp()[mask]-1).abs().max());assert error<1e-6,('Initial replay mismatch',error)
 return {'codes':x,'actions':a,'tasks':task,'valid':mask,'old_logp':old,'advantages':adv,'returns':returns,
  'trials':trials,'proof_error':proof_error,'initial_replay_error':error}

def trial_metric(trial):
 e=trial['episode'];return {'episode_id':e['episode_id'],'scene':e['scene'],'domain':e['domain'],
  'category':e['category'],'distance_group':e['distance_group'],'success':bool(trial['metrics'].get('success',False)),
  'steps':len(trial['steps']),'collisions':sum(s['collision'] for s in trial['steps'])}

try:
 initialize_evaluation()
 atomic(out/'startup_verified.json',{'source_sha256':manifest['warm_start_commit']['checkpoint_sha256'],
  'data':bank.provenance,'fresh_baseline_evaluation':False,'time':time.time()})
 if begin and begin%10==0 and not (out/'evaluations'/f'step_{begin:06d}/summary.json').exists():evaluate(begin)
 with ThreadPoolExecutor(max_workers=8) as pool:
  for update in range(begin+1,args.updates+1):
   started=time.time();batches=[]
   for offset in (0,8):
    batch=[bank.sample(rng,(update-1)*16+offset+i) for i in range(8)]
    assert all(e['scene'] in bank.provenance['train_layouts'] for e in batch)
    batches.append(collect(batch,pool))
   flat=torch.cat([b['advantages'][b['valid']] for b in batches]);mean=flat.mean();std=flat.std(unbiased=False).clamp_min(1e-6)
   total_actions=sum(int(b['valid'].sum()) for b in batches)
   losses=[];value_losses=[];replay_error=0.;gradient_norms=[];module_gradients={};bc_losses=[];clip_fractions=[]
   for epoch in range(2):
    optimizer.zero_grad(set_to_none=True);value_optimizer.zero_grad(set_to_none=True)
    actor_parts=[];critic_parts=[]
    for b in batches:
     result=replay(model,b['codes'],b['tasks'],b['actions'])
     loss,audit=policy_objective(result['probabilities'],b['actions'],b['old_logp'],(b['advantages']-mean)/std,b['valid'])
     if epoch==0:replay_error=max(replay_error,audit['max_ratio_error'])
     clip_fractions.append(audit['clip_fraction'])
     actor_parts.append(loss*int(b['valid'].sum())/total_actions)
     belief=result['beliefs'];goals=b['tasks'][:,None].expand(b['valid'].shape)
     value=critic(belief.flatten(0,1),goals.flatten()).reshape_as(b['returns'])
     critic_parts.append(F.smooth_l1_loss(value[b['valid']],b['returns'][b['valid']])*int(b['valid'].sum())/total_actions)
    if epoch==0:assert replay_error<1e-6,('PPO likelihood mismatch before update',replay_error)
    demos=[episodes[rng.randrange(len(episodes))] for _ in range(8)]
    anchor=demonstration_loss(demos);actor_loss=sum(actor_parts)+.1*anchor;critic_loss=sum(critic_parts)
    assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)
    actor_loss.backward()
    for name,param in model.named_parameters():
     if param.grad is not None:
      key=name.split('.')[0];module_gradients[key]=module_gradients.get(key,0.)+float(param.grad.norm())
    norm=torch.nn.utils.clip_grad_norm_(trainable,1.);assert torch.isfinite(norm)
    critic_loss.backward();value_norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),5.);assert torch.isfinite(value_norm)
    optimizer.step();value_optimizer.step()
    gradient_norms.append(float(norm));losses.append(float(actor_loss.detach()));value_losses.append(float(critic_loss.detach()));bc_losses.append(float(anchor.detach()))
   save(out/'latest',update)
   if update%10==0:save(out/f'step_{update:06d}',update)
   directory=out/'rollouts'/f'step_{update:06d}';directory.mkdir(parents=True,exist_ok=True)
   trials=[t for b in batches for t in b['trials']]
   for i,t in enumerate(trials):atomic(directory/f'{i:03d}.json',t,compact=True)
   record={'update':update,'time':time.time(),'seconds':time.time()-started,'training':aggregate([trial_metric(t) for t in trials]),
    'actor_losses':losses,'value_losses':value_losses,'anchor_losses':bc_losses,'actor_gradient_norms':gradient_norms,
    'actor_module_gradient_norms':module_gradients,'ppo_replay_max_ratio_error':replay_error,
    'symbolic_proof_max_error':max(b['proof_error'] for b in batches),'ppo_clip_fractions':clip_fractions,
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'actions':total_actions,
    'revision_events':sum(len(t['programs'])-1 for t in trials)}
   append(out/'updates.jsonl',record);progress(state='training',**record)
   del batches,result,actor_parts,critic_parts,anchor,actor_loss,critic_loss
   if update%10==0:evaluate(update)
 progress(state='complete',update=args.updates)
except BaseException as error:
 progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
 for sim in sims:sim.close()
'''
(root/'scripts/train_full_automaton_rl_v62.py').write_text(prefix+initial+bc+evaluation+rl)
