"""Repair measured goal erasure in learned categorical state machines.

Independent arms compare recurrent state with reset-state control. No RGB is
loaded while fitting. The original fixed80 is not used for fitting or selection.
"""
import argparse,fcntl,gc,hashlib,json,os,random,sys,time,traceback
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
from nesy_vla.evaluation import aggregate

p=argparse.ArgumentParser();p.add_argument('--arm',choices=['bridge_only','goal_programs'],required=True)
p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=64)
p.add_argument('--evaluate-every',type=int,default=20);args=p.parse_args()
out=base/'goal_conditioned_automaton_v55'/args.arm;out.mkdir(parents=True,exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (out/'status.json').exists() and read(out/'status.json')['state']=='complete':raise SystemExit('Completed; do not rerun')
torch.set_num_threads(2);torch.manual_seed(5100);random.seed(5100);rng=random.Random(5100)
cfg=read(base/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
from stripsnav.io import committed_torch_save,load_committed
source=base/'spatial_automaton_v43';extra=base/'data/onpolicy_recovery_v48'
source_commit=read(source/'COMMITTED');extra_commit=read(extra/'TOKENS_COMMITTED')
assert bank.provenance==source_commit['data_provenance']
assert sha(extra/'visual_tokens.npy')==extra_commit['tokens_sha256']
assert sha(extra/'training_rows.json')==extra_commit['training_rows_sha256']
assert sha(source/'visual_vocabulary.pt')==source_commit['vocabulary_sha256']==extra_commit['vocabulary_sha256']
rows=read(source/'training_rows.json');old_count=len(rows);rows+=read(extra/'training_rows.json')
codes=np.concatenate([np.load(source/'visual_tokens.npy'),np.load(extra/'visual_tokens.npy')])
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
architecture={'patches':49,'vocabulary':64,'tasks':2,'factors':4,'states':16,'actions':17,'width':64,'rank':4,'reset_memory':False,'context_bridge':True,'task_banks':args.arm=='goal_programs'}
model=FactorizedTokenAutomaton(**architecture).cuda()
warm_source=base/'factorized_automaton_v51/recurrent/step_000020'
warm=load_committed(warm_source)
model.initialize_from_shared(warm['model'])
source_model=FactorizedTokenAutomaton(**warm['manifest']['architecture']).cuda().eval()
source_model.load_state_dict(warm['model'])
with torch.no_grad():
 sample=torch.from_numpy(codes[:64].astype(np.int64)).cuda();goals=torch.arange(64,device='cuda')%2
 _,memory=model.initial_state(64,'cuda');prior=torch.full((64,),17,device='cuda',dtype=torch.long)
 before=source_model.step(source_model.generate(goals,sample[:,None],0),sample,memory,prior,0)
 after=model.step(model.generate(goals,sample[:,None],0),sample,memory,prior,0)
 assert torch.equal(before['probabilities'],after['probabilities'])
 assert torch.equal(before['observed_ids'],after['observed_ids'])
 initial_error=float((before['probabilities']-after['probabilities']).abs().max())
del source_model,warm;gc.collect();torch.cuda.empty_cache()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.0001)
manifest={'architecture':architecture,'warm_start_checkpoint':str(warm_source),'warm_start_commit':read(warm_source/'COMMITTED'),'initial_function_max_error':initial_error,'optimizer_resume':False,'source_commit':source_commit,'recovery_commit':read(extra/'COMPLETE'),
 'recovery_tokens':extra_commit,'train_rows_sha256':sha(source/'training_rows.json'),
 'train_tokens_sha256':sha(source/'visual_tokens.npy'),'data':bank.provenance,
 'epochs':args.epochs,'batch_size':args.batch_size,'evaluate_every':args.evaluate_every,
 'parameters':sum(p.numel() for p in model.parameters()),'horizon':32,'revision_interval':10,
 'training_inputs':'Only frozen integer patch codes, task IDs and demonstrated executed actions.',
 'learned_program':'Generated predicate definitions, categorical transition conclusions and action rule conclusions. Revision every10 actions.',
 'rule_semantics':'Categorical equality conjunctions over automatically learned observation/state IDs. No human semantic predicates.',
 'losses':{'action':1.,'observation_reconstruction':.2,'next_discrete_observation':.2,'inverse_action':.1,'state_balance':.005},
 'fixed80_used':False,'oracle_or_geometry_at_inference':False,
 'recovery_supervised_weight':4.,'context_prefix_imitation_weight':0.,
 'teacher_forced_training_accuracy_is_not_navigation_success':True,'goal_adaptation':'Zero-initialized non-saturating goal/history bridge; treatment additionally learns goal-indexed definition/transition/conclusion banks. No predicate meanings or transitions are supplied.'}
if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
else:atomic(out/'manifest.json',manifest)
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
    state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
begin=0
if (out/'latest/COMMITTED').exists():
    saved=load_committed(out/'latest');assert saved['manifest']==manifest
    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);begin=saved['epoch']
    rng.setstate(saved['rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])

def save(folder,epoch):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'manifest':manifest,
             'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    committed_torch_save(folder,payload,{'epoch':epoch,'horizon':32,'revision_interval':10,'optimizer_included':True})

all_codes=torch.from_numpy(codes.astype(np.int64)).cuda()
labels=torch.tensor([r['label'] for r in rows],device='cuda')
supervised=torch.tensor([r['supervised'] for r in rows],device='cuda')
weight=torch.tensor([4. if i>=old_count else 1. for i in range(len(rows))],device='cuda')
histograms=np.stack([np.bincount(codes[:,p],minlength=64) for p in range(49)])
atomic(out/'training_data.json',{'episodes':len(episodes),'frames':len(rows),'supervised':int(supervised.sum()),
 'source_hash':manifest['train_rows_sha256'],'recovery_hash':extra_commit['training_rows_sha256'],
 'per_patch_majority_accuracy':float(histograms.max(1).sum()/histograms.sum()),
 'context_rows_excluded_from_imitation':int((~supervised).sum()),'development_ids':[e['episode_id'] for e in bank.dev]})

def train_batch(batch):
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
        if before_state is not None:
            inverse=model.inverse_action(before_state,r['state_values'])
            inverse_loss=(F.cross_entropy(inverse,y[:,step-1],reduction='none')*active).sum()/denominator
            loss=loss+.1*inverse_loss;counts['inverse_loss']+=float(inverse_loss.detach())
        for distribution in [r['observed_probabilities'],r['state_probabilities']]:
            mean=(distribution*active[:,None,None]).sum(0)/active.sum().clamp_min(1)
            loss=loss+.005*(mean*(mean.clamp_min(1e-12)*model.states).log()).sum(-1).mean()/length
        total=total+loss
        seen+=(r['state_values'].detach()*active[:,None,None]).sum(0)
        previous=r['state_values'];previous_ids=r['state_ids'];last_action=y[:,step];before_state=previous
    assert torch.isfinite(total)
    optimizer.zero_grad(set_to_none=True);total.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);optimizer.step()
    counts.update(loss=float(total.detach()),gradient_norm=float(norm),state_counts=seen.cpu().tolist())
    return counts

first_rows={}
for i,row in enumerate(rows):first_rows.setdefault(row['episode_id'],i)
goal_pairs=[]
for group in bank.paired_demo_groups:
 i,j=[first_rows[eid] for eid in group]
 if rows[i]['label']!=rows[j]['label']:goal_pairs.append((i,j))
assert len(goal_pairs)==55
@torch.no_grad()
def paired_goal_audit():
 inputs=[];goals=[];targets=[]
 for i,j in goal_pairs:
  inputs.extend([i,i]);goals.extend([tasks.index(rows[i]['category']),tasks.index(rows[j]['category'])]);targets.extend([rows[i]['label'],rows[j]['label']])
 x=all_codes[torch.tensor(inputs,device='cuda')];goal=torch.tensor(goals,device='cuda')
 _,prior=model.initial_state(len(x),'cuda');program=model.generate(goal,x[:,None],0)
 result=model.step(program,x,prior,torch.full((len(x),),17,device='cuda',dtype=torch.long),0)
 p=result['probabilities'];actions=p.argmax(-1)
 return {'pairs':len(goal_pairs),'changed_actions':int((actions[::2]!=actions[1::2]).sum()),
  'matched_demonstrations':int((actions==torch.tensor(targets,device='cuda')).sum()),'total_actions':len(targets),
  'mean_probability_total_variation':float((p[::2]-p[1::2]).abs().sum(-1).mean()/2),
  'qualification':'Training-pair goal sensitivity, not navigation success.'}

vision=None;sims=[];vision_cfg=None;vocabulary=None
def initialize_evaluation():
    global vision,vision_cfg,vocabulary,sims
    if vision is not None:return
    from nesy_vla.model import NavigationModel
    loc=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development';vision_cfg=read(loc.parent/'config.json')
    visual=NavigationModel(vision_cfg).eval();load_checkpoint(loc,visual);visual.requires_grad_(False).cuda()
    vision=visual.grounder.vision;del visual;gc.collect()
    vocabulary=torch.load(source/'visual_vocabulary.pt',map_location='cuda',weights_only=True)
    for p in sorted((source/'development').glob('*/trace.json'))[:3]:
        old=read(p);assert encode(Image.open(p.parent/'first.png').convert('RGB')).tolist()==old['steps'][0]['symbols'][:49]
    sims=[bank.simulator(out/f'sim_{j}.log',0,evaluation=True) for j in range(8)]
    atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pids':[s.process.pid for s in sims],'time':time.time()})

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
    for e in bank.dev:
        marker=directory/e['episode_id']/'evaluation.json'
        if marker.exists():done.append(read(marker))
        else:pending.append(e)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(pending),8):
            batch=pending[offset:offset+8];n=len(batch)
            responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(sims,batch)))
            images=[pil(r['rgb']) for r in responses];task=torch.tensor([tasks.index(e['category']) for e in batch],device='cuda')
            memory_ids,memory=model.initial_state(n,'cuda');last_action=torch.full((n,),17,device='cuda',dtype=torch.long)
            active=list(range(n));hist=[];trials=[{'episode':e,'steps':[],'programs':[]} for e in batch]
            for e,image in zip(batch,images):
                folder=directory/e['episode_id'];folder.mkdir(exist_ok=True);image.save(folder/'first.png')
            for step in range(32):
                x=torch.stack([encode(image) for image in images]);hist.append(x)
                if step%10==0:
                    program=model.generate(task,torch.stack(hist[-11:],1),step)
                    for i in active:trials[i]['programs'].append({'start':step,'coefficients':program.coefficients[i].tolist()})
                result=model.step(program,x,memory,last_action,step)
                proof=model.proof(program,result['observed_ids'],memory_ids,last_action,result['state_ids'],step)
                error=float((proof['probabilities']-result['probabilities']).abs().max());proof_error=max(proof_error,error);assert error<1e-7
                chosen=result['probabilities'].argmax(-1)
                for i in active:
                    trials[i]['steps'].append({'step':step,'patch_tokens':x[i].tolist(),'observation_tokens':result['observed_ids'][i].tolist(),
                     'previous_state':memory_ids[i].tolist(),'previous_action':int(last_action[i]),'state':result['state_ids'][i].tolist(),
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
    summary=aggregate(done);atomic(directory/'summary.json',summary)
    atomic(directory/'proof_audit.json',{'maximum_action_probability_error':proof_error,'all_actions_have_discrete_transition_and_rule_proofs':True})
    best=read(out/'best_score.json') if (out/'best_score.json').exists() else {'success_rate':-1}
    if summary['success_rate']>best['success_rate']:save(out/'best_development',epoch);atomic(out/'best_score.json',dict(summary,epoch=epoch))
    model.train();return summary

try:
    if begin and begin%args.evaluate_every==0 and not (out/'evaluations'/f'step_{begin:06d}'/'summary.json').exists():evaluate(begin)
    for epoch in range(begin+1,args.epochs+1):
        started=time.time();order=list(range(len(episodes)));rng.shuffle(order);scores=[]
        for start in range(0,len(order),args.batch_size):
            score=train_batch([episodes[i] for i in order[start:start+args.batch_size]]);scores.append(score)
        record={'epoch':epoch,'time':time.time(),'seconds':time.time()-started,
         'loss':float(np.mean([r['loss'] for r in scores])),
         'teacher_forced_action_accuracy':sum(r['correct'] for r in scores)/sum(r['supervised'] for r in scores),
         'next_token_accuracy':sum(r['prediction_correct'] for r in scores)/max(1,sum(r['prediction_tokens'] for r in scores)),
         'state_ids_used_per_factor':(np.sum([r['state_counts'] for r in scores],axis=0)>0).sum(-1).tolist(),
         'gradient_norm_mean':float(np.mean([r['gradient_norm'] for r in scores])),
         'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
         **{key:float(np.mean([r[key] for r in scores])) for key in ['action_loss','reconstruction_loss','prediction_loss','inverse_loss']}}
        record['paired_goal_audit']=paired_goal_audit()
        save(out/'latest',epoch);append(out/'epochs.jsonl',record);progress(state='training',**record)
        if epoch%args.evaluate_every==0:
            save(out/f'step_{epoch:06d}',epoch);evaluate(epoch)
    progress(state='complete',epoch=args.epochs)
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    for sim in sims:sim.close()
