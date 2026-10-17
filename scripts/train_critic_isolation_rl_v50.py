"""Matched critic-gradient isolation on the corrected automatically learned rules."""
import argparse,copy,fcntl,gc,hashlib,json,os,random,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

base=Path('/home/user/xjy/nesy_vla_20261010');root=Path(__file__).resolve().parent
if not (root/'nesy_vla').exists():root=base/'symbolic_paired_v12'
sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,image_tensor,atomic,append,sha,pil
from nesy_vla.model import NavigationModel
from nesy_vla.evaluation import aggregate
from nesy_vla.reward_experiment import training_reward
from nesy_vla.token_rule_revision import TokenFacts,TokenProgramReviser,convert_v43_trees,PackedRuleProgram

parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=40)
parser.add_argument('--arm',choices=['shared_critic','detached_critic'],required=True)
parser.add_argument('--simulators',type=int,default=8);parser.add_argument('--episodes',type=int,default=16)
args=parser.parse_args();assert args.episodes%args.simulators==0
torch.set_num_threads(2);torch.manual_seed(4500);random.seed(4500);rng=random.Random(4500)
out=base/'critic_isolation_rl_v50'/args.arm;out.mkdir(parents=True,exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'status.json').exists() or read(out/'status.json')['state']!='complete'
source=base/'recovery_rule_learning_v49/without_previous_action';source_commit=read(source/'COMMITTED')
vocabulary_source=base/'spatial_automaton_v43'
assert read(source/'status.json')['state']=='complete'
assert sha(source/'programs.json')==source_commit['program_sha256'] and sha(vocabulary_source/'visual_vocabulary.pt')==source_commit['vocabulary_sha256']
cfg=read(base/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
from stripsnav.io import committed_torch_save,load_committed
schema=(64,)*49+(65,)*50
raw=read(source/'programs.json');tasks=sorted(raw);programs={c:PackedRuleProgram(convert_v43_trees(t,schema),schema) for c,t in raw.items()}
tree_count=len(programs[tasks[0]].roots);del raw;gc.collect()
reviser=TokenProgramReviser(schema,len(tasks),tree_count,detach_value_features=args.arm=='detached_critic')
optimizer=torch.optim.AdamW(reviser.parameters(),lr=3e-4,weight_decay=.01)
vision_source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
vision_cfg=read(vision_source.parent/'config.json');model=NavigationModel(vision_cfg).eval();load_checkpoint(vision_source,model)
model.requires_grad_(False).to('cuda:0');vision=model.grounder.vision;del model;gc.collect()
vocabulary=torch.load(vocabulary_source/'visual_vocabulary.pt',map_location='cuda:0',weights_only=True)
manifest={'source_commit':source_commit,'arm':args.arm,'detach_value_features':args.arm=='detached_critic','previous_action_hidden':True,'data':bank.provenance,'horizon':32,'revision_interval':10,
          'updates':args.updates,'simulators':args.simulators,'episodes_per_update':args.episodes,
          'program_generator_parameters':sum(p.numel() for p in reviser.parameters()),
          'algorithm':'PPO, two replay epochs, ratio clipping0.2, gamma0.99, GAE0.95, learning rate0.0003',
          'train_inputs':'Learned integer visual token history. Prior-action slot always unknown64, matching source rules. Program weights frozen for ten actions.',
          'trainable':'Only the small symbolic program reviser and auxiliary scalar value head; vocabulary and automatically induced Boolean clauses are frozen.',
          'no_manual_predicate_labels':True,'no_map_or_geometry_input':True,'fixed80_used':False,
          'program_change_scope':'Context changes learned rule strengths. New clauses and definitions are not yet synthesized online.',
          'reward_overrides':cfg.get('training_reward_overrides'),
          'development_ids':[e['episode_id'] for e in bank.dev]}
if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
else:atomic(out/'manifest.json',manifest)
state={'state':'initializing','pid':os.getpid(),'time':time.time()};atomic(out/'status.json',state)
def progress(**fields):
    state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
begin=0
if (out/'latest/COMMITTED').exists():
    checkpoint=load_committed(out/'latest');assert checkpoint['source_commit']==source_commit
    reviser.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);begin=checkpoint['update']
    rng.setstate(checkpoint['rng']);torch.set_rng_state(checkpoint['torch_rng'])

@torch.no_grad()
def encode(images):
    codes=[]
    # Preserve the evaluated source's one-image numerical path exactly.
    for image in images:
        x=image_tensor(image,vision_cfg['image_size'],'cuda:0')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=torch.nn.functional.adaptive_avg_pool2d(vision(x),(7,7)).flatten(2).transpose(1,2)
        v=F.normalize(((f.float()-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
        codes.append(torch.cdist(v.flatten(0,1),vocabulary['centers']).argmin(1).cpu().long())
    return torch.stack(codes)

def action_probabilities(weights,evidence):
    p=torch.einsum('r,sra->sa',weights.double(),torch.as_tensor(evidence,dtype=torch.float64))
    if not torch.isfinite(p).all() or (p<0).any() or (p.sum(-1)<=0).any():raise ValueError('No valid rule proof; no fallback')
    return p/p.sum(-1,keepdim=True)

def save(folder,update):
    payload={'model':reviser.state_dict(),'optimizer':optimizer.state_dict(),'update':update,'rng':rng.getstate(),
             'torch_rng':torch.get_rng_state(),'source_commit':source_commit,'manifest':manifest}
    committed_torch_save(folder,payload,{'epoch':update,'revision_interval':10,'horizon':32,'optimizer_included':True})

@torch.no_grad()
def run_batch(simulators,episodes,stochastic,pool):
    responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(simulators,episodes)))
    images=[pil(r['rgb']) for r in responses]
    records=[{'episode':e,'history':[],'chunks':[],'rows':[],'previous':torch.full((49,),64,dtype=torch.long),'last_action':64} for e in episodes]
    active=list(range(len(episodes)))
    for step in range(32):
        codes=encode([images[i] for i in active]);chosen=[]
        for offset,i in enumerate(active):
            record=records[i];facts=torch.cat([codes[offset],record['previous'],torch.tensor([64])])
            record['history'].append(facts.clone());task=tasks.index(episodes[i]['category'])
            if step%10==0:
                history=torch.stack(record['history'][-11:]);revision,value=reviser.generate(task,TokenFacts(history,schema),step)
                chunk={'task':task,'history':history,'start':step,'old_weights':revision.weights.clone(),'old_value':float(value),'rows':[]}
                record['chunks'].append(chunk)
            chunk=record['chunks'][-1]
            evidence,leaves=programs[episodes[i]['category']].evidence(TokenFacts(facts[None],schema))
            p=action_probabilities(chunk['old_weights'],evidence)[0]
            action=int(torch.multinomial(p,1)) if stochastic else int(p.argmax())
            row={'action':action,'old_logp':float(p[action].log()),'value':chunk['old_value'],
                 'evidence':evidence[0].copy(),'leaf_ids':leaves[0].tolist(),'symbols':facts.tolist(),
                 'image_sha256':hashlib.sha256(np.asarray(images[i]).tobytes()).hexdigest(),
                 'probabilities':p.tolist(),'step':step,'program_start':chunk['start']}
            record['rows'].append(row);chunk['rows'].append(row)
            record['previous']=codes[offset].clone();record['last_action']=action;chosen.append(action)
        responses=list(pool.map(lambda pair:simulators[pair[0]].call('step',action=pair[1]),zip(active,chosen)))
        remaining=[]
        for i,response in zip(active,responses):
            adjusted=training_reward(response,cfg.get('training_reward_overrides')) if stochastic else response
            row=records[i]['rows'][-1]
            row.update(reward=float(adjusted['reward']),collision=bool(response.get('feedback',{}).get('collision',False)),
                       reward_terms=copy.deepcopy(adjusted.get('training_reward_audit',{}).get('training_terms',response.get('metrics',{}).get('reward_terms',{}))))
            records[i]['metrics']=copy.deepcopy(response.get('metrics',{}));images[i]=pil(response['rgb'])
            if not response['done']:remaining.append(i)
        active=remaining
        if not active:break
    for record in records:
        carry=0.;next_value=0.
        for row in reversed(record['rows']):
            delta=row['reward']+.99*next_value-row['value'];carry=delta+.99*.95*carry
            row['advantage']=carry;row['return']=carry+row['value'];next_value=row['value']
    return records

def compact(record):
    episode=record['episode'];rows=record['rows']
    return {'episode':episode,'metrics':record['metrics'],
            'programs':[{'start':c['start'],'history':c['history'].tolist(),'weights':c['old_weights'].tolist()} for c in record['chunks']],
            'steps':[{k:v for k,v in r.items() if k!='evidence'} for r in rows]}

def metrics(record):
    episode=record['episode'];rows=record['rows']
    return {'episode_id':episode['episode_id'],'scene':episode['scene'],'category':episode['category'],
            'domain':episode['domain'],'distance_group':episode['distance_group'],'success':bool(record['metrics'].get('success',False)),
            'steps':len(rows),'collisions':sum(r['collision'] for r in rows)}

def evaluate(update,pool):
    directory=out/'evaluations'/f'step_{update:06d}';directory.mkdir(parents=True,exist_ok=True)
    done=[];pending=[]
    for episode in bank.dev:
        marker=directory/episode['episode_id']/'evaluation.json'
        if marker.exists():done.append(read(marker))
        else:pending.append(episode)
    for offset in range(0,len(pending),args.simulators):
        batch=pending[offset:offset+args.simulators]
        trials=run_batch(sims[:len(batch)],batch,False,pool)
        for record in trials:
            folder=directory/record['episode']['episode_id'];folder.mkdir(exist_ok=True)
            atomic(folder/'trace.json',compact(record),compact=True);row=metrics(record);atomic(folder/'evaluation.json',row);done.append(row)
        progress(state='evaluating',update=update,development=aggregate(done))
    summary=aggregate(done);atomic(directory/'summary.json',summary)
    best=read(out/'best_score.json') if (out/'best_score.json').exists() else {'success_rate':-1}
    if summary['success_rate']>best['success_rate']:
        save(out/'best_development',update);atomic(out/'best_score.json',dict(summary,update=update))
    return summary

sims=[]
try:
    # Initial-token fidelity is checked on existing saved frames, with no new rollout.
    checks=[]
    for marker in sorted((source/'development').glob('*/evaluation.json'))[:3]:
        trace=read(marker.parent/'trace.json');image=Image.open(marker.parent/'first.png').convert('RGB')
        observed=encode([image])[0].tolist();assert observed==trace['steps'][0]['symbols'][:49]
        facts=TokenFacts(torch.tensor(trace['steps'][0]['symbols'])[None],schema)
        evidence,_=programs[trace['episode']['category']].evidence(facts)
        p=action_probabilities(torch.ones(tree_count)/tree_count,evidence)[0]
        assert np.max(np.abs(p.numpy()-np.asarray(trace['steps'][0]['probabilities'])))<1e-6
        checks.append(marker.parent.name)
    atomic(out/'initial_fidelity.json',{'existing_frames_checked':checks,'fresh_rollouts':0,'time':time.time()})
    for index in range(args.simulators):sims.append(bank.simulator(out/f'simulator_{index}.log',0,evaluation=True))
    atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pids':[s.process.pid for s in sims],'time':time.time()})
    with ThreadPoolExecutor(max_workers=args.simulators) as pool:
        if begin and begin%10==0 and not (out/'evaluations'/f'step_{begin:06d}'/'summary.json').exists():evaluate(begin,pool)
        for update in range(begin+1,args.updates+1):
            started=time.time();records=[]
            for offset in range(0,args.episodes,args.simulators):
                batch=[bank.sample(rng,(update-1)*args.episodes+offset+i) for i in range(args.simulators)]
                assert all(e['scene'] in bank.provenance['train_layouts'] for e in batch)
                records+=run_batch(sims,batch,True,pool)
            chunks=[c for r in records for c in r['chunks']];rows=[r for record in records for r in record['rows']]
            advantage=torch.tensor([r['advantage'] for r in rows]);mean=float(advantage.mean());std=max(float(advantage.std(unbiased=False)),1e-6)
            replay_error=0;gradient_norm=0;losses=[]
            for epoch in range(2):
                optimizer.zero_grad();total=[]
                for chunk in chunks:
                    revision,value=reviser.generate(chunk['task'],TokenFacts(chunk['history'],schema),chunk['start'])
                    p=action_probabilities(revision.weights,np.stack([r['evidence'] for r in chunk['rows']]))
                    actions=torch.tensor([r['action'] for r in chunk['rows']]);old=torch.tensor([r['old_logp'] for r in chunk['rows']],dtype=torch.float64)
                    logp=p[torch.arange(len(actions)),actions].log();ratio=(logp-old).exp()
                    if epoch==0:replay_error=max(replay_error,float((ratio.detach()-1).abs().max()))
                    adv=torch.tensor([(r['advantage']-mean)/std for r in chunk['rows']]);returns=torch.tensor([r['return'] for r in chunk['rows']])
                    policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
                    value_loss=F.smooth_l1_loss(value.expand_as(returns),returns)
                    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1).mean()
                    rule_kl=(revision.weights*(revision.weights.clamp_min(1e-30)*tree_count).log()).sum()
                    total.append((policy+.5*value_loss-.01*entropy+.005*rule_kl)*len(actions)/len(rows))
                loss=torch.stack(total).sum();assert torch.isfinite(loss)
                if epoch==0:assert replay_error<1e-6,('Same-program likelihood mismatch',replay_error)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(reviser.parameters(),1.)
                assert torch.isfinite(norm);gradient_norm=float(norm);optimizer.step();losses.append(float(loss.detach()))
            save(out/'latest',update)
            if update%10==0:save(out/f'step_{update:06d}',update)
            directory=out/'rollouts'/f'step_{update:06d}';directory.mkdir(parents=True,exist_ok=True)
            for index,record in enumerate(records):atomic(directory/f'{index:03d}.json',compact(record),compact=True)
            summary=aggregate([metrics(r) for r in records]);record={'time':time.time(),'update':update,'seconds':time.time()-started,
                'training':summary,'losses':losses,'gradient_norm':gradient_norm,'ppo_replay_max_ratio_error':replay_error,
                'revision_events':sum(len(r['chunks'])-1 for r in records),'actions':len(rows)}
            append(out/'updates.jsonl',record);progress(state='training',**record)
            if update%10==0:evaluate(update,pool)
        progress(state='complete',update=args.updates)
except BaseException as error:
    progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    for sim in sims:sim.close()
