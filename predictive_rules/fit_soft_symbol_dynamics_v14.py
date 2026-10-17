"""Fit discrete state distributions from demonstrations and measured branches.

Definitions and goal generation stay frozen. This is a bounded training-only
candidate; the exact symbolic runtime and observation boundary are unchanged.
"""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld,Config
from predictive_nesy.subgoal_data import TransitionBank,load_windows,batch
from predictive_nesy.data import sha

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'pilot_soft_symbol_dynamics_v14'
assert not out.exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}[7]<300
out.mkdir();torch.set_num_threads(2);torch.manual_seed(3414);began=time.time()
path=ROOT/'run_definition_world_v12/checkpoint_001600.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
assert sha(path)==json.loads(path.with_suffix('.COMMITTED.json').read_text())['sha256']
m=DefinitionConditionedWorld(Config(**saved['model_config'])).cuda().eval();m.load_state_dict(saved['model']);m.requires_grad_(False)
fixed={k:v.detach().cpu().clone() for k,v in m.state_dict().items() if not k.startswith('definition_rules.')}
programs={k:[] for k in ['definitions','context']};rows={k:[] for k in ['program','source','target','mask']}
episode_ids=[];kind=[];program_count=0

def add_program(p):
    global program_count
    n=len(p['definitions']);ids=torch.arange(program_count,program_count+n,device='cuda');program_count+=n
    programs['definitions'].append(p['definitions'].detach().cpu())
    programs['context'].append(m.context_norm(p['context_raw']).detach().cpu())
    return ids

def add_rows(program,source,target,mask,names,row_kind):
    for k,v in dict(program=program,source=source,target=target,mask=mask).items():rows[k].append(v.detach().cpu())
    episode_ids.extend(names);kind.extend([row_kind]*len(names))

with torch.no_grad():
    for bank_index,item in enumerate(saved['manifest']['transition_banks']):
        bank=TransitionBank(item['path'],saved['manifest']['data']['train_layouts'])
        for (start,offset),group in sorted(bank.by_key.items()):
            for pos in range(0,len(group),64):
                selected=group[pos:pos+64];samples=[]
                for name,t in selected:
                    d=bank.episodes[name];begin=max(0,start-10)
                    samples.append(dict(task=int(d['task']),history=d['codes'][begin:start+1],
                        previous_actions=np.array([int(d['actions'][j-1]) if j else 17 for j in range(begin,start+1)]),
                        prefix=d['codes'][start:t+1],successors=d['successors'][t]))
                d={k:torch.as_tensor(np.asarray([s[k] for s in samples]),device='cuda').long() for k in samples[0]}
                lengths=torch.full((len(samples),),d['history'].shape[1],device='cuda',dtype=torch.long)
                p=m.generate(d['task'],d['history'],d['previous_actions'],lengths,start=start)
                observed,memory,previous,codes=m.encode_sequence(p,d['prefix']);source=observed[:,-1].argmax(-1)
                features=m.observation(m.embed(d['successors'])).flatten(0,1)
                memory,features,_=m.advance(memory.repeat_interleave(17,0),previous.repeat_interleave(17,0),
                    codes.repeat_interleave(17,0),features,d['successors'].flatten(0,1))
                features=F.normalize(features+m.memory_projection(memory),dim=-1)
                defs=F.normalize(p['definitions'],dim=-1).repeat_interleave(17,0)
                target=(8*torch.einsum('bd,bkd->bk',features,defs).float()).softmax(-1).reshape(len(samples),17,128)
                add_rows(add_program(p),source,target,torch.ones((len(samples),17),device='cuda'),[name for name,t in selected],0)
        print(json.dumps(dict(event='cached_bank',bank=bank_index,states=len(episode_ids))),flush=True)
    windows,provenance=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
    for start in sorted({w['start'] for w in windows}):
        selected=[w for w in windows if w['start']==start]
        for pos in range(0,len(selected),64):
            subset=selected[pos:pos+64];d=batch(subset,'cuda');width=min(start+1,11)
            p=m.generate(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start)
            observed=m.training_targets(p,d['targets'],d['actions']).squeeze(2)
            bi,ti=d['action_valid'].nonzero(as_tuple=True);n=len(bi);actions=d['actions'][bi,ti]
            target=torch.zeros((n,17,128),device='cuda');target[torch.arange(n,device='cuda'),actions]=observed[bi,ti+1]
            mask=F.one_hot(actions,17).float();ids=add_program(p)
            add_rows(ids[bi],observed[bi,ti].argmax(-1),target,mask,[subset[i]['episode_id'] for i in bi.tolist()],1)
    print(json.dumps(dict(event='cache_complete',states=len(episode_ids),programs=program_count)),flush=True)
programs={k:torch.cat(v).cuda() for k,v in programs.items()};data={k:torch.cat(v).cuda() for k,v in rows.items()}
held=np.asarray([int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%5==0 for name in episode_ids]);kind=np.asarray(kind)
assert not(set(np.asarray(episode_ids)[~held])&set(np.asarray(episode_ids)[held]))
selections={(split,k):torch.tensor(np.flatnonzero((held if split=='held' else ~held)&(kind==k)),device='cuda')
    for split in ['fit','held'] for k in [0,1]}
assert all(len(v)>100 for v in selections.values())

def logits(ix):
    pi=data['program'][ix];defs=programs['definitions'][pi];context=programs['context'][pi];source=data['source'][ix];head=m.definition_rules
    keys=F.normalize(head.key(defs),dim=-1);chosen=defs[torch.arange(len(ix),device='cuda'),source]
    x=head.premise(chosen)[:,None]+head.action.weight[None]+head.context(context)[:,None]
    query=F.normalize(head.query(F.gelu(x)),dim=-1)
    return head.log_scale.exp().clamp(1,64)*torch.einsum('bad,bkd->bak',query,keys).float()

with torch.no_grad():
    ix=torch.arange(8,device='cuda');pi=data['program'][ix]
    complete=m.definition_rules(programs['definitions'][pi],programs['context'][pi])[torch.arange(8,device='cuda'),data['source'][ix]]
    error=float((complete-logits(ix).softmax(-1)).abs().max());assert error<2e-6

def evaluate(split,k):
    selection=selections[split,k];correct=torch.zeros(17,device='cuda');count=torch.zeros_like(correct);ce=0.;entropy=0.
    with torch.no_grad():
        for pos in range(0,len(selection),256):
            ix=selection[pos:pos+256];logprob=logits(ix).log_softmax(-1);target=data['target'][ix];mask=data['mask'][ix]
            correct+=((logprob.argmax(-1)==target.argmax(-1))*mask).sum(0);count+=mask.sum(0)
            ce+=float((-(target*logprob).sum(-1)*mask).sum())
            entropy+=float((-(target*target.clamp_min(1e-8).log()).sum(-1)*mask).sum())
    total=float(count.sum())
    return dict(states=len(selection),transitions=int(total),accuracy=float(correct.sum())/total,
        movement_accuracy=float(correct[1:].sum()/count[1:].sum().clamp_min(1)),
        stop_accuracy=float(correct[0]/count[0].clamp_min(1)),kl=(ce-entropy)/total)

def all_metrics():return {split:{['branches','demonstrations'][k]:evaluate(split,k) for k in [0,1]} for split in ['fit','held']}
initial=all_metrics();report=dict(training_only=True,no_navigation_run=True,source=str(path),source_sha256=sha(path),
    objective='Soft categorical successor-state cross entropy only; no continuous-state or image prediction',
    target_source='Frozen learned posterior over128 anonymous states; actual future images are training labels only',
    source_premise='Actual categorical predecessor state for world-model supervision only; runtime propagates its own states',
    trainable='definition_rules only',frozen='grounding, definition generator, goal head and all boundary encoders',
    heldout_scope='Additional-fit episode split, not unseen validation: the source already trained on these data',
    demo_branch_batch_sizes=[384,256],row_weighting='Mean per row over its available actions,60% demonstration rows and40% counterfactual rows',
    full_row_error=error,initial=initial,history=[])
print(json.dumps(dict(event='initial',**report)),flush=True)
for parameter in m.definition_rules.parameters():parameter.requires_grad_(True)
opt=torch.optim.AdamW(m.definition_rules.parameters(),lr=3e-4,weight_decay=1e-4)
best=sum(initial['held'][k]['kl'] for k in ['branches','demonstrations']);best_update=0
best_weights={k:v.detach().cpu().clone() for k,v in m.definition_rules.state_dict().items()};train_start=time.time()
for step in range(1,1501):
    ix=torch.cat([selections['fit',k][torch.randint(len(selections['fit',k]),(n,),device='cuda')] for k,n in [(1,384),(0,256)]])
    logprob=logits(ix).log_softmax(-1);mask=data['mask'][ix]
    loss=((-(data['target'][ix]*logprob).sum(-1)*mask).sum(-1)/mask.sum(-1)).mean()
    opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.definition_rules.parameters(),1.);assert torch.isfinite(norm);opt.step()
    if step%100==0:print(json.dumps(dict(event='training',update=step,loss=float(loss.detach()),seconds=time.time()-train_start)),flush=True)
    if step%300==0:
        score=all_metrics();report['history'].append(dict(update=step,metrics=score));criterion=sum(score['held'][k]['kl'] for k in ['branches','demonstrations'])
        if criterion<best:
            best=criterion;best_update=step;best_weights={k:v.detach().cpu().clone() for k,v in m.definition_rules.state_dict().items()}
        print(json.dumps(dict(event='metrics',**report['history'][-1])),flush=True)
m.definition_rules.load_state_dict(best_weights)
assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in fixed.items())
report.update(complete=True,best_additional_update=best_update,selected_metrics=all_metrics(),all_nonrule_weights_unchanged=True,
    seconds=time.time()-began,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,navigation_improvement_proven=False)
manifest=dict(saved['manifest'],soft_symbol_fit=dict(source=str(path),source_sha256=sha(path),updates=best_update,demonstrations_included=True,
    generated_parameters_unchanged=True,continuous_state_prediction=False))
candidate=out/'candidate.pt';torch.save(dict(model=m.cpu().state_dict(),model_config=m.export_config(),manifest=manifest,update=saved['update']+best_update),candidate)
candidate.with_suffix('.COMMITTED.json').write_text(json.dumps(dict(path=candidate.name,sha256=sha(candidate),update=saved['update']+best_update,time=time.time()),indent=2))
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(event='complete',**report)),flush=True)
