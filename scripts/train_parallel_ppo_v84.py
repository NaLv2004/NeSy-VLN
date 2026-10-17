"""Learn generated soft finite-state programs from integer-token demonstrations.

No manual predicate definitions, no map, no continuous visual action bypass.
"""
import argparse,datetime,fcntl,gc,hashlib,json,os,random,sys,time,traceback
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

base=Path('/home/user/xjy/nesy_vla_20261010');root=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(1,str(base/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,append,setup,DataBank,image_tensor,pil,load_checkpoint
from nesy_vla.factorized_automaton import FactorizedTokenAutomaton
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.evaluation import aggregate

from nesy_vla.automaton_policy_gradient import SymbolicValue,replay,advantages,policy_objective,configure_trainable
from nesy_vla.reward_experiment import training_reward
from nesy_vla.parallel_rollout import collect_parallel

p=argparse.ArgumentParser();p.add_argument('--arm',choices=['joint','action_rules_only'],required=True)
p.add_argument('--updates',type=int,default=120);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
assert args.arm=='joint','One primary experiment only'
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(0);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
out=base/'parallel_ppo_v84';out=out/'preflight' if args.verify_only else out;rank_dir=out/f'rank{rank}';rank_dir.mkdir(parents=True,exist_ok=True)
lock=(rank_dir/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'status.json').exists() or read(out/'status.json')['state']!='complete'
torch.set_num_threads(2);torch.manual_seed(5100);random.seed(5100);rng=random.Random(5100+1009*rank)
cfg=read(base/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
from stripsnav.io import load_committed
from nesy_vla.small_checkpoint import save_small_checkpoint
from nesy_vla.distributed_symbolic import global_moments,average_gradients,gather_objects,assert_identical_parameters
from nesy_vla.sequence_perception import SequencePerception
from nesy_vla.observation_inverse import observation_inverse_logits
def committed_torch_save(folder,payload,metadata):
    return save_small_checkpoint(folder,payload,metadata,reservation_root=base/'parallel_ppo_v84')
source=base/'perceptual_sequences_v73/tail_1e4';extra=base/'data/onpolicy_recovery_v48'
source_commit=read(source/'TOKENS_COMMITTED');extra_commit=read(extra/'COMPLETE')
assert bank.provenance==source_commit['data_provenance']
assert sha(source/'visual_tokens.npy')==source_commit['tokens_sha256']
assert sha(source/'training_rows.json')==source_commit['training_rows_sha256']
assert sha(source_commit['vocabulary_path'])==source_commit['vocabulary_sha256']
rows=read(source/'training_rows.json');old_count=source_commit['old_count']
codes=np.load(source/'visual_tokens.npy')
assert codes.shape==(len(rows),49) and codes.min()>=0 and codes.max()<64

groups={}
for i,row in enumerate(rows):groups.setdefault(row['episode_id'],[]).append(i)
heldout={e['episode_id'] for e in bank.dev+bank.fixed}
assert not heldout.intersection(groups)
assert not heldout.intersection(r.get('source_episode_id',r['episode_id']) for r in rows)
tasks=['chair','table'];episodes=[]
for eid,indices in groups.items():
    assert len(indices)<=32 and indices==list(range(indices[0],indices[-1]+1))
    assert rows[indices[-1]]['label']==0
    episodes.append({'id':eid,'task':tasks.index(rows[indices[0]]['category']),'indices':indices})

warm_source=base/'parallel_ppo_v84/source_snapshot'
assert read(warm_source/'COMMITTED')==read(base/'parallel_ppo_v84/SOURCE.json')['commit']
warm=load_committed(warm_source);architecture=warm['manifest']['architecture']
assert warm['manifest']['source_commit']==source_commit
assert not architecture['reset_memory'] and warm['manifest']['data']==bank.provenance
model=ProbabilisticTokenAutomaton(**architecture).cuda();model.load_state_dict(warm['model']);parent_training={k:warm[k] for k in ['optimizer','critic','value_optimizer']}
parent_epoch=warm['epoch'];parent_rng=warm['rank_rng'][rank];del warm
trainable=configure_trainable(model,args.arm)
critic=SymbolicValue(model.factors,model.states,model.tasks).cuda()
optimizer=torch.optim.AdamW(trainable,lr=3e-5,weight_decay=.0001)
value_optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-3,weight_decay=.0001)
optimizer.load_state_dict(parent_training['optimizer']);critic.load_state_dict(parent_training['critic']);value_optimizer.load_state_dict(parent_training['value_optimizer']);del parent_training
manifest={'architecture':architecture,'arm':args.arm,'warm_start_checkpoint':str(warm_source),
 'warm_start_commit':read(warm_source/'COMMITTED'),'data':bank.provenance,'horizon':32,'revision_interval':10,
 'updates':args.updates,'episodes_per_update':24,'canonical_symbolic_batch':8,'world_size':world,'global_episodes_per_update':24*world,'simultaneous_per_gpu':24,'parallel_collection':'Three canonical8 policy batches; all24 simulators advance concurrently; exact RGB token cache.',
 'actor_lr':3e-5,'critic_lr':1e-3,'ppo_epochs':2,'clip':.2,'gamma':.99,'gae_lambda':.95,
 'entropy_weight':.01,'demonstration_anchor':.1,'auxiliary_critic_gradient_to_actor':False,
 'train_rows_sha256':sha(source/'training_rows.json'),'recovery_commit':extra_commit,
 'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
 'reward_overrides':cfg.get('training_reward_overrides'),'fixed80_used':False,
 'source_development':read(out/'SOURCE.json')['development_before_parallel_change'],
 'action_path':'Integer visual vocabulary -> learned integer observation IDs -> explicit probabilistic finite-state transitions -> pairwise equality rules -> action distribution. Program revisions every10 actions use integer token history only.',
 'comparison':'No control arms. All GPUs jointly optimize one actor, grounding, definitions, transitions and dynamic-program generator.',
 'optimizer_resume':True,'source_sft_optimizer_retained':True,'source_commit':source_commit,
 'selection':'Continue latest committed v82 eightGPU PPO model/optimizer/critic/allrankRNG. Only parallel rollout organization changes; no new baseline or fixed80 evaluation.', 'parent_ppo_update':parent_epoch,
 'inverse_auxiliary':'Preserve parent observed-ID inverse supervision; no recurrent action-bearing belief inverse shortcut.',
 'storage':'Small full-resume atomic checkpoints with 512MiB plus twice serialized size reserve.'}
if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
elif rank==0:atomic(out/'manifest.json',manifest)
dist.barrier()
state={'rank':rank,'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());atomic(rank_dir/'status.json',state);print(json.dumps(state),flush=True)
torch.manual_seed(6100+rank*1009)
begin=parent_epoch
rng.setstate(parent_rng['python']);torch.set_rng_state(parent_rng['torch']);torch.cuda.set_rng_state(parent_rng['cuda']);del parent_rng
if (out/'latest/COMMITTED').exists():
 saved=load_committed(out/'latest');assert saved['manifest']==manifest
 model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
 critic.load_state_dict(saved['critic']);value_optimizer.load_state_dict(saved['value_optimizer']);begin=saved['epoch']
 state_rng=saved['rank_rng'][rank];rng.setstate(state_rng['python']);torch.set_rng_state(state_rng['torch']);torch.cuda.set_rng_state(state_rng['cuda'])
 del saved
assert_identical_parameters(model);assert_identical_parameters(critic)

def save(folder,epoch):
 rngs=gather_objects({'python':rng.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()})
 if rank==0:
  payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'critic':critic.state_dict(),
    'value_optimizer':value_optimizer.state_dict(),'epoch':epoch,'manifest':manifest,'rank_rng':rngs,
    'parent_source':str(warm_source)}
  committed_torch_save(folder,payload,{'epoch':epoch,'horizon':32,'revision_interval':10,'optimizer_included':True,'ranks':world})
 dist.barrier()
all_codes=torch.from_numpy(codes.astype(np.int64)).cuda()
labels=torch.tensor([r['label'] for r in rows],device='cuda')
supervised=torch.tensor([r['supervised'] for r in rows],device='cuda')
weight=torch.tensor([4. if i>=old_count else 1. for i in range(len(rows))],device='cuda')
histograms=np.stack([np.bincount(codes[:,p],minlength=64) for p in range(49)])
atomic(rank_dir/'training_data.json',{'episodes':len(episodes),'frames':len(rows),'supervised':int(supervised.sum()),
 'source_hash':manifest['train_rows_sha256'],'recovery_hash':extra_commit['training_rows_sha256'],
 'per_patch_majority_accuracy':float(histograms.max(1).sum()/histograms.sum()),
 'context_rows_excluded_from_imitation':int((~supervised).sum()),'development_ids':[e['episode_id'] for e in bank.dev]})

def demonstration_loss(batch):
    length=max(len(e['indices']) for e in batch);n=len(batch)
    indices=torch.zeros(n,length,dtype=torch.long,device='cuda');valid=torch.zeros_like(indices,dtype=torch.bool)
    for j,e in enumerate(batch):
        k=len(e['indices']);indices[j,:k]=torch.tensor(e['indices'],device='cuda');valid[j,:k]=True
    x=all_codes[indices];y=labels[indices];mask=valid&supervised[indices]
    weights=weight[indices]*mask;task=torch.tensor([e['task'] for e in batch],device='cuda')
    previous_ids,previous=model.initial_state(n,'cuda');last_action=torch.full((n,),17,device='cuda',dtype=torch.long)
    total=x.new_zeros((),dtype=torch.float);counts={k:0. for k in ['action_loss','reconstruction_loss','prediction_loss','inverse_loss','correct','supervised','prediction_correct','prediction_tokens']}
    seen=torch.zeros(model.factors,model.states,device='cuda');before_state=None
    for step in range(length):
        if step%10==0:program=model.generate(task,x[:,max(0,step-10):step+1],step)
        r=model.step(program,x[:,step],previous,last_action,step)
        active=valid[:,step].float();denominator=valid.sum().clamp_min(1)
        action=-r['probabilities'].clamp_min(1e-20).log().gather(1,y[:,step,None]).squeeze(1)
        action_loss=(action*weights[:,step]).sum()/weights.sum().clamp_min(1)
        reconstruction=F.cross_entropy(r['reconstruction'].transpose(1,2),x[:,step],reduction='none').mean(-1)
        reconstruct_loss=(reconstruction*active).sum()/denominator
        loss=action_loss+.2*reconstruct_loss
        counts['correct']+=float(((r['probabilities'].argmax(-1)==y[:,step])&mask[:,step]).sum())
        counts['supervised']+=float(mask[:,step].sum());counts['action_loss']+=float(action_loss.detach())
        counts['reconstruction_loss']+=float(reconstruct_loss.detach())
        if step+1<length:
            next_valid=valid[:,step+1].float();prediction=model.next_observation(r['state_values'],y[:,step])
            pred=F.cross_entropy(prediction.transpose(1,2),x[:,step+1],reduction='none').mean(-1)
            pred_loss=(pred*next_valid).sum()/denominator;loss=loss+.2*pred_loss
            counts['prediction_loss']+=float(pred_loss.detach())
            counts['prediction_correct']+=float(((prediction.argmax(-1)==x[:,step+1])*next_valid[:,None]).sum())
            counts['prediction_tokens']+=float(next_valid.sum()*49)
        if step:
            inverse=observation_inverse_logits(model,program,x[:,step-1],x[:,step],step)
            inverse_loss=(F.cross_entropy(inverse,y[:,step-1],reduction='none')*active).sum()/denominator
            loss=loss+.1*inverse_loss
        for distribution in [r['observed_probabilities'],r['state_probabilities']]:
            mean=(distribution*active[:,None,None]).sum(0)/active.sum().clamp_min(1)
            loss=loss+.005*(mean*(mean.clamp_min(1e-12)*model.states).log()).sum(-1).mean()/length
        total=total+loss
        seen+=(F.one_hot(r['state_ids'],model.states).float()*active[:,None,None]).sum(0)
        previous=r['state_values'];previous_ids=r['state_ids'];last_action=y[:,step];before_state=previous
    assert torch.isfinite(total)
    return total
vision=None;sims=[];vision_cfg=None;vocabulary=None;prefix=None;perception=None
def initialize_evaluation(launch_simulators=True):
    global vision,vision_cfg,vocabulary,sims,prefix,perception
    if vision is not None:return
    from nesy_vla.model import NavigationModel
    location=Path(source_commit['encoder_source']);vision_cfg=read(location.parent/'config.json')
    assert read(location/'COMMITTED')==source_commit['encoder_source_commit']
    visual=NavigationModel(vision_cfg).eval();load_checkpoint(location,visual);visual.requires_grad_(False).cuda()
    vision=visual.grounder.vision;del visual
    if hasattr(vision,'backbone'):vision=vision.backbone
    assert isinstance(vision,torch.nn.Sequential) and len(vision)==8
    component=Path(source_commit['component_checkpoint']);assert read(component/'COMMITTED')==source_commit['component_commit']
    payload=load_committed(component);vision[6:].load_state_dict(payload['visual_tail']);del payload;gc.collect()
    vocabulary=torch.load(source_commit['vocabulary_path'],map_location='cuda',weights_only=True)
    # The v80 parent kept the identical v73 frontend; reuse the existing,
    # hash-verified integer corpus instead of recomputing 24,848 images.
    for i in [0,len(rows)//2,len(rows)-1]:
        with Image.open(rows[i]['path']) as image:assert torch.equal(encode(image.convert('RGB')),all_codes[i])
    if launch_simulators:
        sims=[bank.simulator(rank_dir/f'sim_{j}.log',0,evaluation=True) for j in range(24)]
        atomic(rank_dir/'processes.json',{'pid':os.getpid(),'simulator_pids':[s.process.pid for s in sims],'time':time.time()})

@torch.no_grad()
def encode(image):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        f=F.adaptive_avg_pool2d(vision(image_tensor(image,vision_cfg['image_size'],'cuda')),(7,7)).flatten(2).transpose(1,2)
    z=F.normalize(((f.float()-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
    return torch.cdist(z.flatten(0,1),vocabulary['centers']).argmin(1)

@torch.no_grad()
def evaluate(epoch):
    initialize_evaluation();model.eval();directory=out/'evaluations'/f'step_{epoch:06d}'
    directory.mkdir(parents=True,exist_ok=True);done=[];pending=[];proof_error=0.
    for i,e in enumerate(bank.dev):
        if (i//8)%world!=rank:continue
        marker=directory/e['episode_id']/'evaluation.json'
        if marker.exists():done.append(read(marker))
        else:pending.append(e)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(pending),8):
            batch=pending[offset:offset+8];n=len(batch)
            responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(sims,batch)))
            images=[pil(r['rgb']) for r in responses];images+=images[:1]*(8-n);task=torch.tensor([tasks.index(e['category']) for e in batch+[batch[0]]*(8-n)],device='cuda')
            memory_ids,memory=model.initial_state(8,'cuda');last_action=torch.full((8,),17,device='cuda',dtype=torch.long)
            active=list(range(n));hist=[];trials=[{'episode':e,'steps':[],'programs':[]} for e in batch]
            for e,image in zip(batch,images):
                folder=directory/e['episode_id'];folder.mkdir(exist_ok=True);image.save(folder/'first.png')
            for step in range(32):
                x=torch.stack([encode(image) for image in images]);hist.append(x)
                if step%10==0:
                    program=model.generate(task,torch.stack(hist[-11:],1),step)
                    for i in active:trials[i]['programs'].append({'start':step,'coefficients':program.coefficients[i].tolist()})
                result=model.step(program,x,memory,last_action,step)
                proof=model.proof(program,result['observed_ids'],memory,last_action,result['state_values'],step)
                error=float((proof['probabilities']-result['probabilities']).abs().max());proof_error=max(proof_error,error);assert error<1e-7
                chosen=result['probabilities'].argmax(-1)
                for i in active:
                    trials[i]['steps'].append({'step':step,'patch_tokens':x[i].tolist(),'observation_tokens':result['observed_ids'][i].tolist(),
                     'previous_state':memory_ids[i].tolist(),'previous_action':int(last_action[i]),'state':result['state_ids'][i].tolist(),
                     'previous_belief':memory[i].tolist(),'state_belief':result['state_values'][i].tolist(),
                     'action':int(chosen[i]),'probabilities':result['probabilities'][i].tolist(),
                     'transition_probabilities':proof['transition_probabilities'][i].tolist(),'rule_evidence':proof['rule_evidence'][i].tolist(),
                     'program_start':program.start,'image_sha256':hashlib.sha256(np.asarray(images[i]).tobytes()).hexdigest()})
                memory=result['state_values'];memory_ids=result['state_ids'];last_action=chosen
                responses=list(pool.map(lambda pair:sims[pair[0]].call('step',action=pair[1]),[(i,int(chosen[i])) for i in active]))
                remaining=[]
                for i,response in zip(active,responses):
                    trials[i]['steps'][-1]['collision']=bool(response.get('feedback',{}).get('collision',False))
                    trials[i]['metrics']=response.get('metrics',{});images[i]=pil(response['rgb'])
                    if not response['done']:remaining.append(i)
                active=remaining
                if not active:break
            for trial,image in zip(trials,images):
                e=trial['episode'];folder=directory/e['episode_id'];image.save(folder/'last.png');atomic(folder/'trace.json',trial,compact=True)
                row={'episode_id':e['episode_id'],'scene':e['scene'],'domain':e['domain'],'category':e['category'],'distance_group':e['distance_group'],
                     'success':bool(trial['metrics'].get('success',False)),'steps':len(trial['steps']),'collisions':sum(r['collision'] for r in trial['steps'])}
                atomic(folder/'evaluation.json',row);done.append(row)
            progress(state='evaluating',epoch=epoch,development=aggregate(done),symbolic_proof_max_error=proof_error)
    done=[row for part in gather_objects(done) for row in part]
    assert len(done)==80 and len({r['episode_id'] for r in done})==80
    summary=aggregate(done)
    flags=torch.tensor([proof_error],device='cuda');dist.all_reduce(flags,op=dist.ReduceOp.MAX)
    assert float(flags[0])==0
    if rank==0:
        atomic(directory/'summary.json',summary)
        atomic(directory/'proof_audit.json',{'maximum_action_probability_error':float(flags[0]),'all_actions_have_discrete_transition_and_rule_proofs':True,'episodes':80})
    dist.barrier()
    best=read(out/'best_score.json') if (out/'best_score.json').exists() else {'success_rate':-1}
    if summary['success_rate']>best['success_rate']:
        save(out/'best_development',epoch)
        if rank==0:atomic(out/'best_score.json',dict(summary,epoch=epoch))
    dist.barrier();model.train();return summary

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
 initialize_evaluation(launch_simulators=not args.verify_only)
 atomic(rank_dir/'startup_verified.json',{'source_sha256':manifest['warm_start_commit']['checkpoint_sha256'],
  'data':bank.provenance,'fresh_baseline_evaluation':False,'time':time.time()})
 # The parent checkpoint was already evaluated when appropriate; do not replay it.
 with ThreadPoolExecutor(max_workers=24) as pool:
  for update in range(begin+1,args.updates+1):
   started=time.time()
   batch=[bank.sample(rng,parent_epoch*128+(update-parent_epoch-1)*24*world+rank*24+i) for i in range(24)]
   assert all(e['scene'] in bank.provenance['train_layouts'] for e in batch)
   batches,collection=collect_parallel(batch,pool,model=model,critic=critic,sims=sims,encode=encode,cfg=cfg,task_names=tasks)
   optimization_started=time.time()
   flat=torch.cat([b['advantages'][b['valid']] for b in batches]);mean,std=global_moments(flat)
   total_actions=sum(int(b['valid'].sum()) for b in batches)
   global_actions=torch.tensor(total_actions,device='cuda');dist.all_reduce(global_actions)
   rank_weight=world*total_actions/int(global_actions)
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
    anchor=demonstration_loss(demos);actor_loss=rank_weight*sum(actor_parts)+.1*anchor;critic_loss=rank_weight*sum(critic_parts)
    assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)
    actor_loss.backward()
    for name,param in model.named_parameters():
     if param.grad is not None:
      key=name.split('.')[0];module_gradients[key]=module_gradients.get(key,0.)+float(param.grad.norm())
    average_gradients(trainable)
    norm=torch.nn.utils.clip_grad_norm_(trainable,1.);assert torch.isfinite(norm)
    critic_loss.backward();average_gradients(critic.parameters());value_norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),5.);assert torch.isfinite(value_norm)
    optimizer.step();value_optimizer.step()
    gradient_norms.append(float(norm));losses.append(float(actor_loss.detach()));value_losses.append(float(critic_loss.detach()));bc_losses.append(float(anchor.detach()))
   synchronization_error=assert_identical_parameters(model);assert_identical_parameters(critic)
   save(out/'latest',update)
   if update%10==0:save(out/f'step_{update:06d}',update)
   directory=out/'rollouts'/f'step_{update:06d}'/f'rank{rank}';directory.mkdir(parents=True,exist_ok=True)
   trials=[t for b in batches for t in b['trials']]
   for i,t in enumerate(trials):atomic(directory/f'{i:03d}.json',t,compact=True)
   global_metrics=[row for part in gather_objects([trial_metric(t) for t in trials]) for row in part]
   assert len(global_metrics)==24*world
   record={'collection_timing':collection,'optimization_and_commit_seconds':time.time()-optimization_started,'rank':rank,'world':world,'synchronization_error':synchronization_error,'global_actions':int(global_actions),'update':update,'time':time.time(),'seconds':time.time()-started,'training':aggregate(global_metrics),
    'actor_losses':losses,'value_losses':value_losses,'anchor_losses':bc_losses,'actor_gradient_norms':gradient_norms,
    'actor_module_gradient_norms':module_gradients,'ppo_replay_max_ratio_error':replay_error,
    'symbolic_proof_max_error':max(b['proof_error'] for b in batches),'ppo_clip_fractions':clip_fractions,
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'actions':total_actions,
    'revision_events':sum(len(t['programs'])-1 for t in trials)}
   append(rank_dir/'updates.jsonl',record);progress(state='training',**record)
   del batches,result,actor_parts,critic_parts,anchor,actor_loss,critic_loss
   if update%10==0:evaluate(update)
 progress(state='complete',update=args.updates)
except Exception as error:
 progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
 for sim in sims:sim.close()
 dist.destroy_process_group()
