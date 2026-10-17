"""One bounded training-only probe: fit actual learned-state transitions.

This updates only the existing generated-rule decoder. It introduces no value,
reward, semantic state, map, actor or runtime observation correction. Future
images supply labels after execution. The model still exports complete tables.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld,Config
from predictive_nesy.subgoal_data import TransitionBank
from predictive_nesy.data import sha

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
p.add_argument('--updates',type=int,default=1500);p.add_argument('--train-context',action='store_true');a=p.parse_args()
out=(root/a.out).resolve();assert out.parent==root and out.name.startswith('pilot_')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in used.splitlines()}[7]<300,'GPU7 occupied; do not disturb it'
assert not out.exists(),'Never overwrite a prior probe'
out.mkdir();torch.set_num_threads(2);torch.manual_seed(2137)
checkpoint=root/a.checkpoint;saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert sha(checkpoint)==json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())['sha256']
if a.train_context:
    from predictive_nesy.learned_rule_context import LearnedRuleContextWorld
    m=LearnedRuleContextWorld(Config(**saved['model_config'])).cuda().eval()
    incompatible=m.load_state_dict(saved['model'],strict=False)
    assert not incompatible.unexpected_keys and all(n.startswith('rule_context.') for n in incompatible.missing_keys)
    m.reset_rule_context()
else:
    m=DefinitionConditionedWorld(Config(**saved['model_config'])).cuda().eval();m.load_state_dict(saved['model'])
m.requires_grad_(False)
trainable_prefixes=('definition_rules.','rule_context.') if a.train_context else ('definition_rules.',)
fixed={n:v.cpu().clone() for n,v in m.state_dict().items() if not n.startswith(trainable_prefixes)}
began=time.time();cache={k:[] for k in ['definitions','context','source','target']};ids=[];bank_ids=[]
if a.train_context:cache.update({k:[] for k in ['history','previous','task','length']})

def logits_rows(definitions,context,source):
    head=m.definition_rules
    keys=F.normalize(head.key(definitions),dim=-1)
    chosen=definitions[torch.arange(len(source),device=source.device),source]
    x=head.premise(chosen)[:,None]+head.action.weight[None]+head.context(context)[:,None]
    query=F.normalize(head.query(F.gelu(x)),dim=-1)
    return head.log_scale.exp().clamp(1,64)*torch.einsum('bad,bkd->bak',query,keys).float()

with torch.no_grad():
    for bindex,item in enumerate(saved['manifest']['transition_banks']):
        bank=TransitionBank(item['path'],saved['manifest']['data']['train_layouts'])
        for (start,offset),group in sorted(bank.by_key.items()):
            for pos in range(0,len(group),64):
                selected=group[pos:pos+64];rows=[]
                for name,t in selected:
                    d=bank.episodes[name];begin=max(0,start-10)
                    rows.append(dict(task=int(d['task']),history=d['codes'][begin:start+1],
                        previous_actions=np.array([int(d['actions'][j-1]) if j else 17 for j in range(begin,start+1)]),
                        prefix=d['codes'][start:t+1],successors=d['successors'][t]))
                data={k:torch.as_tensor(np.asarray([r[k] for r in rows]),device='cuda').long() for k in rows[0]}
                lengths=torch.full((len(rows),),data['history'].shape[1],device='cuda',dtype=torch.long)
                program=m.generate(data['task'],data['history'],data['previous_actions'],lengths,start=start)
                observed,memory,previous,codes=m.encode_sequence(program,data['prefix']);source=observed[:,-1].argmax(-1)
                future=m.observation(m.embed(data['successors'])).flatten(0,1)
                memory,features,_=m.advance(memory.repeat_interleave(17,0),previous.repeat_interleave(17,0),
                    codes.repeat_interleave(17,0),future,data['successors'].flatten(0,1))
                features=F.normalize(features+m.memory_projection(memory),dim=-1)
                defs=F.normalize(program['definitions'],dim=-1).repeat_interleave(17,0)
                target=torch.einsum('bd,bkd->bk',features,defs).argmax(-1).reshape(len(rows),17)
                context=m.context_norm(program['context_raw'])
                for key,value in dict(definitions=program['definitions'],context=context,source=source,target=target).items():cache[key].append(value.detach().cpu())
                if a.train_context:
                    length=data['history'].shape[1]
                    history=F.pad(data['history'],(0,0,0,11-length))
                    previous=F.pad(data['previous_actions'],(0,11-length),value=17)
                    for key,value in dict(history=history,previous=previous,task=data['task'],length=lengths).items():cache[key].append(value.cpu())
                ids.extend(name for name,t in selected);bank_ids.extend([bindex]*len(rows))
        print(json.dumps(dict(event='cache_bank',bank=bindex,states=len(ids),seconds=time.time()-began)),flush=True)
cache={k:torch.cat(v).cuda() for k,v in cache.items()};bank_ids=np.asarray(bank_ids)
held=np.asarray([int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%5==0 for name in ids])
fit=torch.as_tensor(np.flatnonzero(~held),device='cuda');hold=torch.as_tensor(np.flatnonzero(held),device='cuda')
assert len(fit)>1000 and len(hold)>200
assert not(set(np.asarray(ids)[~held])&set(np.asarray(ids)[held]))
with torch.no_grad():
    select=torch.arange(8,device='cuda');defs=cache['definitions'][select];context=cache['context'][select];source=cache['source'][select]
    full=m.definition_rules(defs,context)[torch.arange(8,device='cuda'),source]
    partial=logits_rows(defs,context,source).softmax(-1)
    row_error=float((full-partial).abs().max());assert row_error<2e-6
    if a.train_context:
        learned_context=m.rule_context(cache['task'][select],cache['history'][select],cache['previous'][select],cache['length'][select])
        context_error=float((learned_context-context).abs().max());assert context_error<2e-5
        altered=cache['history'][select].clone();altered_previous=cache['previous'][select].clone()
        for i,length in enumerate(cache['length'][select].tolist()):
            altered[i,length:]=torch.randint(64,altered[i,length:].shape,device='cuda')
            altered_previous[i,length:]=torch.randint(18,altered_previous[i,length:].shape,device='cuda')
        future=m.rule_context(cache['task'][select],altered,altered_previous,cache['length'][select])
        assert torch.equal(future,learned_context),'Padded/future frames affect boundary context'

def get_logits(ix):
    context=(m.rule_context(cache['task'][ix],cache['history'][ix],cache['previous'][ix],cache['length'][ix])
        if a.train_context else cache['context'][ix])
    return logits_rows(cache['definitions'][ix],context,cache['source'][ix])

def metrics(selection):
    correct=0;loss=0.;total=0;by_action=torch.zeros(17,device='cuda')
    with torch.no_grad():
        for pos in range(0,len(selection),256):
            ix=selection[pos:pos+256];logits=get_logits(ix);target=cache['target'][ix]
            mask=logits.argmax(-1)==target;correct+=int(mask.sum());total+=mask.numel();by_action+=mask.sum(0)
            loss+=float(F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='sum'))
    return dict(accuracy=correct/total,correct=correct,branches=total,loss=loss/total,per_action_accuracy=(by_action/len(selection)).cpu().tolist())

report=dict(source=str(checkpoint),source_sha256=sha(checkpoint),training_only=True,
    heldout_scope='Episodes excluded from this additional fitting only; the source checkpoint has already trained on these banks. This is not unseen navigation validation.',
    fit_states=len(fit),held_states=len(hold),total_states=len(ids),full_row_probability_error=row_error,
    trainable=list(trainable_prefixes),objective='Categorical cross entropy for all17 actions at actually observed learned states; no planning imitation gradients',
    manual_semantic_labels=False,goal_head_frozen=True,representation_frozen=True,no_navigation_run=True,
    initial=dict(fit=metrics(fit),held=metrics(hold)),history=[])
print(json.dumps(dict(event='initial',**report)),flush=True)
for name,v in m.named_parameters():v.requires_grad_(name.startswith(trainable_prefixes))
if a.train_context:
    m.rule_context.train()
    m.rule_context.history.flatten_parameters()
parameters=[v for v in m.parameters() if v.requires_grad]
opt=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=1e-4)
def selected_weights():return {n:v.detach().cpu().clone() for n,v in m.state_dict().items() if n.startswith(trainable_prefixes)}
best=report['initial']['held']['accuracy'];best_step=0;best_weights=selected_weights()
train_start=time.time()
for step in range(1,a.updates+1):
    ix=fit[torch.randint(len(fit),(512,),device='cuda')]
    logits=get_logits(ix)
    loss=F.cross_entropy(logits.flatten(0,1),cache['target'][ix].flatten());assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm);opt.step()
    if step==1 or step%50==0:print(json.dumps(dict(event='training',update=step,loss=float(loss.detach()),seconds=time.time()-train_start)),flush=True)
    if step%300==0 or step==a.updates:
        score=dict(update=step,fit=metrics(fit),held=metrics(hold),seconds=time.time()-train_start)
        report['history'].append(score);print(json.dumps(dict(event='metrics',**score)),flush=True)
        if score['held']['accuracy']>best:
            best=score['held']['accuracy'];best_step=step;best_weights=selected_weights()
    (out/'status.json').write_text(json.dumps(dict(state='training',update=step,time=time.time())))
state=m.state_dict();state.update(best_weights);m.load_state_dict(state)
assert all(torch.equal(v,m.state_dict()[n].cpu()) for n,v in fixed.items())
report.update(complete=True,best_additional_update=best_step,best_held_accuracy=best,seconds=time.time()-began,
    all_nondynamics_weights_unchanged=True,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,navigation_improvement_proven=False)
manifest=dict(saved['manifest'],world_fit_probe=dict(source=str(checkpoint),additional_updates=best_step,
    source_sha256=sha(checkpoint),no_planner_gradient=True,heldout_scope=report['heldout_scope']))
if a.train_context:manifest['architecture']='anonymous_rule_context_world_v13'
path=out/'candidate.pt';torch.save(dict(model=m.cpu().state_dict(),model_config=m.export_config(),manifest=manifest,update=saved['update']+best_step),path)
path.with_suffix('.COMMITTED.json').write_text(json.dumps(dict(path=path.name,sha256=sha(path),update=saved['update']+best_step,time=time.time()),indent=2))
(out/'report.json').write_text(json.dumps(report,indent=2));(out/'status.json').write_text(json.dumps(dict(state='complete',update=a.updates,time=time.time())))
print(json.dumps(dict(event='complete',**report)),flush=True)
