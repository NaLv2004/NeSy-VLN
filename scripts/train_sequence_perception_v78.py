"""Learn generated soft finite-state programs from integer-token demonstrations.

No manual predicate definitions, no map, no continuous visual action bypass.
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
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.evaluation import aggregate

p=argparse.ArgumentParser();p.add_argument('--arm',choices=['frozen','joint'],required=True)
p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch-size',type=int,default=32)
p.add_argument('--evaluate-every',type=int,default=5);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
out=base/'sequence_perception_v78'/args.arm;out.mkdir(parents=True,exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (out/'status.json').exists() and read(out/'status.json')['state']=='complete':raise SystemExit('Completed; do not rerun')
torch.set_num_threads(2);torch.manual_seed(5100);random.seed(5100);rng=random.Random(5100)
cfg=read(base/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
from stripsnav.io import load_committed
from nesy_vla.small_checkpoint import save_small_checkpoint
from nesy_vla.learned_visual_tokens import TrainableAlphabetAutomaton, SymbolTokens
from nesy_vla.sequence_perception import SequencePerception
from nesy_vla.model import NavigationModel
def committed_torch_save(folder,payload,metadata):
    return save_small_checkpoint(folder,payload,metadata,reservation_root=base/'sequence_perception_v78')
source=base/'perceptual_sequences_v73'/'tail_1e4';extra=base/'data/onpolicy_recovery_v48'
source_commit=read(source/'TOKENS_COMMITTED')
assert bank.provenance==source_commit['data_provenance']
assert sha(source/'visual_tokens.npy')==source_commit['tokens_sha256']
assert sha(source/'training_rows.json')==source_commit['training_rows_sha256']
assert sha(source_commit['vocabulary_path'])==source_commit['vocabulary_sha256']
rows=read(source/'training_rows.json');old_count=source_commit['old_count']
codes=np.load(source/'visual_tokens.npy')
extra_commit={'training_rows_sha256':source_commit['recovery_rows_sha256']}
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
architecture={'patches':49,'vocabulary':64,'tasks':2,'factors':4,'states':16,'actions':17,'width':64,'rank':4,'reset_memory':False,'context_bridge':True,'task_banks':True}
model=TrainableAlphabetAutomaton(**architecture).cuda()
warm_source=base/'perceptual_fsm_v74'/'tail_1e4'/'step_000010'
assert read(warm_source/'COMMITTED')['checkpoint_sha256']=='039086a4e972c7016fce81e226c6c42c71a3b3fd697b9dad3f045917daad71cd'
warm=load_committed(warm_source)
assert architecture==warm['manifest']['architecture']
model.load_state_dict(warm['model'],strict=True)
del warm;gc.collect();torch.cuda.empty_cache()
location=Path(source_commit['encoder_source']);vision_cfg=read(location.parent/'config.json')
assert read(location/'COMMITTED')==source_commit['encoder_source_commit']
visual=NavigationModel(vision_cfg).eval();load_checkpoint(location,visual);visual.requires_grad_(False).cuda()
encoder=visual.grounder.vision;del visual
if hasattr(encoder,'backbone'):encoder=encoder.backbone
assert isinstance(encoder,torch.nn.Sequential) and len(encoder)==8
component=Path(source_commit['component_checkpoint']);assert read(component/'COMMITTED')==source_commit['component_commit']
perception_payload=load_committed(component);encoder[6:].load_state_dict(perception_payload['visual_tail']);del perception_payload
prefix=torch.nn.Sequential(*list(encoder.children())[:7],torch.nn.Sequential(*list(encoder[7].children())[:-1])).eval().requires_grad_(False)
vocabulary=torch.load(source_commit['vocabulary_path'],map_location='cuda',weights_only=True)
perception=SequencePerception(encoder[7][-1],vocabulary,canonical_batch=16).cuda()
perception.block.requires_grad_(args.arm=='joint');del encoder;gc.collect()
groups=[{'params':model.parameters(),'lr':1e-4}]
if args.arm=='joint':groups.append({'params':perception.block.parameters(),'lr':1e-5})
optimizer=torch.optim.AdamW(groups,weight_decay=.0001)
manifest={'architecture':architecture,'warm_start_checkpoint':str(warm_source),'warm_start_commit':read(warm_source/'COMMITTED'),
 'optimizer_resume':False,'source_commit':source_commit,'recovery_commit':read(extra/'COMPLETE'),
 'train_rows_sha256':sha(source/'training_rows.json'),'train_tokens_sha256':sha(source/'visual_tokens.npy'),'data':bank.provenance,
 'epochs':args.epochs,'batch_size':args.batch_size,'evaluate_every':args.evaluate_every,'parameters':sum(p.numel() for p in model.parameters()),
 'horizon':32,'revision_interval':10,'training_inputs':'Only hard integer patch codes, task IDs and demonstrated executed actions.',
 'learned_program':'Small history model generates learned predicate definitions, soft transition tables and action conclusions every10 actions.',
 'rule_semantics':'Conjunctions of equalities over anonymous automatically learned observation/state IDs. No human predicate semantics.',
 'losses':{'action':1.,'observation_reconstruction':.2,'next_discrete_observation':.2,'inverse_action':0.,'state_balance':.005},
 'fixed80_used':False,'oracle_or_geometry_at_inference':False,'recovery_supervised_weight':4.,'context_prefix_imitation_weight':0.,
 'teacher_forced_training_accuracy_is_not_navigation_success':True,
 'state_semantics':'Hard observed predicate IDs, explicit soft beliefs over finite learned state IDs, exact rule-based transition marginalization.',
 'perception':'Matched canonical last-block batch16 in both arms. Frozen prefix cached only in private GPU RAM. Joint arm trains final ConvNeXt block through hard tokens over complete trajectories.',
 'comparison':'Same immutable v74tail epoch10 model, v72 visual tail, seed, original training rows, batch32 and AdamW1e-4. Frozen control versus final-visual-block learning at1e-5 through hard tokens. Both inverse auxiliaries disabled. No baseline navigation rerun.',
 'canonical_last_block_batch':16,'model_lr':1e-4,'visual_lr':1e-5 if args.arm=='joint' else 0.,
 'inverse_arm':'disabled','visual_parameters':sum(p.numel() for p in perception.block.parameters()),
 'checkpoint_policy':'Complete model/optimizer/RNG atomic commits with512MiB+2xactualserializedsize reservation andshared experiment lock.'}
if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
else:atomic(out/'manifest.json',manifest)
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
    state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
begin=0
if (out/'latest/COMMITTED').exists():
    saved=load_committed(out/'latest');assert saved['manifest']==manifest
    model.load_state_dict(saved['model']);perception.block.load_state_dict(saved['visual_last_block']);optimizer.load_state_dict(saved['optimizer']);begin=saved['epoch']
    rng.setstate(saved['rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])

def save(folder,epoch):
    payload={'model':model.state_dict(),'visual_last_block':perception.block.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'manifest':manifest,
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

# Frozen features are confined to this visual frontend. No feature tensor is
# handed to the state-machine generator, transition interpreter or action rules.
prefix_cache=torch.empty((len(rows),768,7,7),device='cuda',dtype=torch.bfloat16)
cached=np.zeros(len(rows),dtype=bool)
@torch.no_grad()
def cache_prefix(indices):
    for i in indices:
        i=int(i)
        if cached[i]:continue
        with Image.open(rows[i]['path']) as image:pixel=image_tensor(image.convert('RGB'),vision_cfg['image_size'],'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):f=prefix(pixel)
        assert f.shape==(1,768,7,7)
        prefix_cache[i].copy_(f[0]);cached[i]=True
        if int(cached.sum())%1024==0:progress(state='caching_frozen_prefix',completed=int(cached.sum()),total=len(rows))

def tokens_for_indices(indices):
    unique,inverse=torch.unique(indices.flatten(),sorted=True,return_inverse=True)
    cache_prefix(unique.tolist())
    tokens=perception(prefix_cache[unique])
    return SymbolTokens(tokens.ids[inverse].reshape(*indices.shape,49),tokens.values[inverse].reshape(*indices.shape,49,64))

def train_batch(batch):
    length=max(len(e['indices']) for e in batch);n=len(batch)
    indices=torch.zeros(n,length,dtype=torch.long,device='cuda');valid=torch.zeros_like(indices,dtype=torch.bool)
    for j,e in enumerate(batch):
        k=len(e['indices']);indices[j,:k]=torch.tensor(e['indices'],device='cuda');valid[j,:k]=True
    x=tokens_for_indices(indices);y=labels[indices];mask=valid&supervised[indices]
    weights=weight[indices]*mask;task=torch.tensor([e['task'] for e in batch],device='cuda')
    previous_ids,previous=model.initial_state(n,'cuda');last_action=torch.full((n,),17,device='cuda',dtype=torch.long)
    total=x.ids.new_zeros((),dtype=torch.float);counts={k:0. for k in ['action_loss','reconstruction_loss','prediction_loss','inverse_loss','correct','supervised','prediction_correct','prediction_tokens']}
    seen=torch.zeros(model.factors,model.states,device='cuda');before_state=None
    for step in range(length):
        if step%10==0:program=model.generate(task,x[:,max(0,step-10):step+1],step)
        r=model.step(program,x[:,step],previous,last_action,step)
        active=valid[:,step].float();denominator=valid.sum().clamp_min(1)
        action=-r['probabilities'].clamp_min(1e-20).log().gather(1,y[:,step,None]).squeeze(1)
        action_loss=(action*weights[:,step]).sum()/weights.sum().clamp_min(1)
        reconstruction=F.cross_entropy(r['reconstruction'].transpose(1,2),x.ids[:,step],reduction='none').mean(-1)
        reconstruct_loss=(reconstruction*active).sum()/denominator
        loss=action_loss+.2*reconstruct_loss
        counts['correct']+=float(((r['probabilities'].argmax(-1)==y[:,step])&mask[:,step]).sum())
        counts['supervised']+=float(mask[:,step].sum());counts['action_loss']+=float(action_loss.detach())
        counts['reconstruction_loss']+=float(reconstruct_loss.detach())
        if step+1<length:
            next_valid=valid[:,step+1].float();prediction=model.next_observation(r['state_values'],y[:,step])
            pred=F.cross_entropy(prediction.transpose(1,2),x.ids[:,step+1],reduction='none').mean(-1)
            pred_loss=(pred*next_valid).sum()/denominator;loss=loss+.2*pred_loss
            counts['prediction_loss']+=float(pred_loss.detach())
            counts['prediction_correct']+=float(((prediction.argmax(-1)==x.ids[:,step+1])*next_valid[:,None]).sum())
            counts['prediction_tokens']+=float(next_valid.sum()*49)
        if args.verify_only:
            with torch.no_grad():
                plain_program=model.generate(task,x.ids[:,max(0,program.start-10):program.start+1],program.start)
                plain=model.step(plain_program,x.ids[:,step],previous.detach(),last_action,step)
                assert torch.equal(plain['probabilities'],r['probabilities'])
                proof=model.proof(program,r['observed_ids'],previous,last_action,r['state_values'],step)
                assert torch.equal(proof['probabilities'],r['probabilities'])
        for distribution in [r['observed_probabilities'],r['state_probabilities']]:
            mean=(distribution*active[:,None,None]).sum(0)/active.sum().clamp_min(1)
            loss=loss+.005*(mean*(mean.clamp_min(1e-12)*model.states).log()).sum(-1).mean()/length
        total=total+loss
        seen+=(F.one_hot(r['state_ids'],model.states).float()*active[:,None,None]).sum(0)
        previous=r['state_values'];previous_ids=r['state_ids'];last_action=y[:,step];before_state=previous
    assert torch.isfinite(total)
    optimizer.zero_grad(set_to_none=True);total.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
    visual_norm=torch.nn.utils.clip_grad_norm_(perception.block.parameters(),1.) if args.arm=='joint' else torch.tensor(0.)
    assert torch.isfinite(visual_norm)
    if args.arm=='joint':assert visual_norm>0,'No visual gradient through sequence rules'
    optimizer.step()
    counts.update(loss=float(total.detach()),gradient_norm=float(norm),visual_gradient_norm=float(visual_norm),state_counts=seen.cpu().tolist())
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
 x=tokens_for_indices(torch.tensor(inputs,device='cuda'));goal=torch.tensor(goals,device='cuda')
 _,prior=model.initial_state(len(x),'cuda');program=model.generate(goal,x[:,None],0)
 result=model.step(program,x,prior,torch.full((len(x),),17,device='cuda',dtype=torch.long),0)
 p=result['probabilities'];actions=p.argmax(-1)
 return {'pairs':len(goal_pairs),'changed_actions':int((actions[::2]!=actions[1::2]).sum()),
  'matched_demonstrations':int((actions==torch.tensor(targets,device='cuda')).sum()),'total_actions':len(targets),
  'mean_probability_total_variation':float((p[::2]-p[1::2]).abs().sum(-1).mean()/2),
  'qualification':'Training-pair goal sensitivity, not navigation success.'}

sims=[]
@torch.no_grad()
def encode(image):
    with torch.autocast('cuda',dtype=torch.bfloat16):features=prefix(image_tensor(image,vision_cfg['image_size'],'cuda'))
    return perception(features).ids[0]

def initialize_evaluation(launch_simulators=True):
    for index in [0,len(rows)//2,len(rows)-1]:
        cache_prefix([index])
        with Image.open(rows[index]['path']) as image:online=encode(image.convert('RGB'))
        offline=tokens_for_indices(torch.tensor([index],device='cuda')).ids[0]
        assert torch.equal(online,offline),'Cached and online perception disagree'
    atomic(out/'perception_replay_verified.json',{'canonical_visual_prefix_batch':1,'canonical_last_block_batch':16,
        'training_rows_checked':3,'exact_integer_token_match':True,'features_stay_inside_perception':True})
    if launch_simulators and not sims:
        sims.extend(bank.simulator(out/f'sim_{j}.log',0,evaluation=True) for j in range(8))
        atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pids':[s.process.pid for s in sims],'time':time.time()})

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
    summary=aggregate(done);atomic(directory/'summary.json',summary)
    atomic(directory/'proof_audit.json',{'maximum_action_probability_error':proof_error,'all_actions_have_discrete_transition_and_rule_proofs':True,'state_semantics':'Soft beliefs over finite learned state IDs; exact transition marginalization and product-conjunction action rules.'})
    best=read(out/'best_score.json') if (out/'best_score.json').exists() else {'success_rate':-1}
    if summary['success_rate']>best['success_rate']:save(out/'best_development',epoch);atomic(out/'best_score.json',dict(summary,epoch=epoch))
    model.train();return summary

try:
    if args.verify_only:
        initialize_evaluation(launch_simulators=False)
        selected=sorted(episodes,key=lambda e:len(e['indices']),reverse=True)[:2]
        x=tokens_for_indices(torch.tensor([e['indices'][0] for e in selected],device='cuda'))
        task=torch.tensor([e['task'] for e in selected],device='cuda')
        _,prior=model.initial_state(len(x),'cuda')
        previous_action=torch.full((len(x),),17,device='cuda',dtype=torch.long)
        program=model.generate(task,x[:,None],0)
        result=model.step(program,x,prior,previous_action,0)
        proof=model.proof(program,result['observed_ids'],prior,previous_action,result['state_values'],0)
        proof_error=float((proof['probabilities']-result['probabilities']).abs().max())
        assert proof_error==0
        del program,result,proof
        score=train_batch(selected)
        assert score['gradient_norm']>0 and score['supervised']>0
        atomic(out/'integration_verified.json',{'time':time.time(),'arm':args.arm,
            'actual_training_episode_ids':[e['id'] for e in selected],
            'episode_lengths':[len(e['indices']) for e in selected],
            'gradient_norm':score['gradient_norm'],'visual_gradient_norm':score['visual_gradient_norm'],'loss':score['loss'],
            'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'integer_action_proof_max_error':proof_error,
            'perception_token_equality':True,'optimizer_step_completed':True,
            'checkpoint_saved':False,'navigation_evaluation_run':False})
        print(json.dumps(read(out/'integration_verified.json')),flush=True)
        raise SystemExit(0)
    cache_prefix(range(len(rows)))
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
         'visual_gradient_norm_mean':float(np.mean([r['visual_gradient_norm'] for r in scores])),
         'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
         **{key:float(np.mean([r[key] for r in scores])) for key in ['action_loss','reconstruction_loss','prediction_loss','inverse_loss']}}
        record['paired_goal_audit']=paired_goal_audit()
        save(out/'latest',epoch);append(out/'epochs.jsonl',record);progress(state='training',**record)
        if epoch%args.evaluate_every==0:
            save(out/f'step_{epoch:06d}',epoch);evaluate(epoch)
    progress(state='complete',epoch=args.epochs)
except Exception as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    for sim in sims:sim.close()
