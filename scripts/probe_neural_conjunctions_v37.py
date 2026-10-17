"""Matched train-only study: learned weighting of generated conjunction proofs."""
import collections,fcntl,json,os,random,sys,time
from pathlib import Path
import torch
from PIL import Image
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,DataBank,image_tensor,atomic,sha
from nesy_vla.model import NavigationModel
from nesy_vla.symbols import tokenize
from nesy_vla.outcome_supervision import SymbolicExecutionDecoder,execution_loss

torch.set_num_threads(2);device='cuda:0';assert torch.cuda.is_available()
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
delta=base/'execution_learning_v34/execution_feedback_diagnostic_weights.pt'
assert read(delta.parent/'result.json')['state']=='complete'
cfg=read(source.parent/'config.json');cfg['recovery_demonstration_root']=str(base/'data/recovery_demonstrations_v27')
cfg['execution_outcome_root']='/home/user/xjy/strips_wm_navigation_20261009/data/search_cost_supervision_v3'
setup(cfg);bank=DataBank(cfg);out=base/'neural_conjunction_learning_v37';out.mkdir(exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (out/'result.json').exists():raise RuntimeError('Inspect existing run before any repeat')
result={'state':'extracting','pid':os.getpid(),'time':time.time(),'source_commit':read(source/'COMMITTED'),
    'initial_delta':str(delta),'initial_delta_sha256':sha(delta),'provenance':bank.provenance,
    'original_development_or_fixed_used':False,'not_navigation_success':True,
    'all_actions_via_generated_conjunctions':True,'no_manual_predicates':True,
    'gate_inputs':'Hard Boolean facts and discrete rule/literal identities only',
    'gate_output':'Scalar per rule, multiplying existing proof; no action head',
    'max_steps':800,'batch_size':32,'learning_rate':2e-4,'arms':{}}
atomic(out/'result.json',result)
dataset=read(base/'execution_learning_v34/dataset.json');selected=[]
lookup={(eid,row['image']):row for eid in bank.recovery_demo_ids for row in bank.demos[eid] if 'execution_outcome_labels' in row}
for original in dataset['examples']:
    row=lookup[(original['id'],original['image'])]
    assert row['action']==original['label'] and row['execution_outcome_labels']==original['outcomes']
    selected.append(dict(original,row=row))
atomic(out/'dataset.json',dataset)
initial=NavigationModel(cfg).eval();load_checkpoint(source,initial);initial.to(device)
with torch.no_grad():
    for start in range(0,len(selected),32):
        batch=selected[start:start+32]
        images=[Image.open(bank.demo_image_path(e['row'])).convert('RGB') for e in batch]
        tokens=initial.grounder.encode(torch.cat([image_tensor(im,cfg['image_size'],device) for im in images])).detach()
        for row,token in zip(batch,tokens):row['tokens']=token
del initial;torch.cuda.empty_cache()
train=[r for r in selected if not r['heldout']];validation=[r for r in selected if r['heldout']]
assert len(train)==609 and len(validation)==155
positive=torch.tensor(dataset['positive_weights'],device=device)

def forward(model,rows):
    probabilities=[];actions=[];predicted=[];labels=[];codes=[];gate_values=[]
    for category in ['chair','table']:
        chosen=[r for r in rows if r['category']==category]
        if not chosen:continue
        p=model.generator('Find the '+category+' and stop close to it.',[],[],None)
        tokens=torch.stack([r['tokens'] for r in chosen])
        h=model.grounder.query(p.definitions[None].expand(len(chosen),-1,-1),tokens)
        state=tokenize(model.grounder.truth(h).squeeze(-1).sigmoid())
        probabilities.append(model.logic.probabilities_for_states(p,state));codes.append(state.codes)
        gate_values.append(model.logic.action_log_gates(p,state).exp())
        predicted.append(model.execution_decoder(p.definitions,state))
        actions.extend(r['label'] for r in chosen);labels.extend(r['outcomes'] for r in chosen)
    return (torch.cat(probabilities),torch.tensor(actions,device=device),torch.cat(predicted),
        torch.tensor(labels,dtype=torch.float32,device=device),torch.cat(codes),torch.cat(gate_values))

@torch.no_grad()
def measure(model,rows):
    predictions=[];actions=[];outputs=[];labels=[];states=[];gates=[];nll=0.
    for start in range(0,len(rows),64):
        p,y,logits,target,codes,g=forward(model,rows[start:start+64])
        predictions.extend(p.argmax(1).tolist());actions.extend(y.tolist());outputs.append(logits);labels.append(target);states.append(codes);gates.append(g)
        nll+=float(-p.gather(1,y[:,None]).clamp_min(1e-9).log().sum())
    output=torch.cat(outputs);truth=torch.cat(labels).bool();est=output>0;summary={}
    for channel,name in enumerate(['collision','no_movement']):
        t=truth[:,:,channel];e=est[:,:,channel];tp=int((e&t).sum());tn=int((~e&~t).sum());pos=int(t.sum());neg=t.numel()-pos
        summary[name]={'balanced_accuracy':.5*(tp/max(1,pos)+tn/max(1,neg))}
    move=[(i,a) for i,a in enumerate(predictions) if a>0];g=torch.cat(gates)
    return {'examples':len(rows),'expert_action_accuracy':sum(a==b for a,b in zip(predictions,actions))/len(rows),
        'turn_action_recall':sum(a==b and b not in [0,4,11] for a,b in zip(predictions,actions))/max(1,sum(b not in [0,4,11] for b in actions)),
        'greedy_counts':dict(collections.Counter(predictions)),'nll':nll/len(rows),'chosen_moves':len(move),
        'chosen_move_observed_collision_rate':sum(bool(truth[i,a-1,0]) for i,a in move)/max(1,len(move)),
        'chosen_move_no_movement_rate':sum(bool(truth[i,a-1,1]) for i,a in move)/max(1,len(move)),
        'auxiliary':summary,'unique_symbol_states':len(torch.unique(torch.cat(states),dim=0)),
        'rule_gate_range':[float(g.min()),float(g.max())]}

result.update(state='training',frames=len(selected),train_frames=len(train),holdout_frames=len(validation),time=time.time());atomic(out/'result.json',result)
for name,enabled in [('conjunction_control',False),('neural_rule_weighting',True)]:
    torch.manual_seed(3701);model=NavigationModel(cfg).eval();load_checkpoint(source,model)
    model.execution_decoder=SymbolicExecutionDecoder(cfg['program']['d'],cfg['program']['a'])
    assert sha(delta)==result['initial_delta_sha256']
    mismatch=model.load_state_dict(torch.load(delta,map_location='cpu',weights_only=True),strict=False)
    assert not mismatch.unexpected_keys and all(k.startswith(('grounder.vision.','teacher.')) for k in mismatch.missing_keys)
    model.logic.enable_action_gates();model.generator.head.neural_conjunction_gates=enabled
    model.grounder.vision.requires_grad_(False);model.grounder.projection.requires_grad_(False);model.grounder.position.requires_grad_(False)
    model.to(device);params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=2e-4,weight_decay=.01);rng=random.Random(3701)
    arm={'initial':{'train':measure(model,train),'holdout':measure(model,validation)},'learning':[]}
    result['arms'][name]=arm;start=time.time()
    if enabled:
        control=result['arms']['conjunction_control']['initial']['holdout']
        assert arm['initial']['holdout']==control,'Initial actions/grounding must be identical'
    for step in range(1,801):
        batch=rng.sample(train,32);p,y,output,target,codes,_=forward(model,batch)
        bc=-p.gather(1,y[:,None]).clamp_min(1e-9).log().mean();aux=execution_loss(output,target,positive)
        loss=bc+.5*aux;optimizer.zero_grad();loss.backward()
        if step==1:
            gate_grad=sum(float(v.grad.detach().square().sum()) for v in model.logic.action_rule_gate.parameters() if v.grad is not None)**.5
            arm['initial_gate_gradient_norm']=gate_grad
            assert (gate_grad>0)==enabled
        norm=torch.nn.utils.clip_grad_norm_(params,2.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite proof gradient')
        optimizer.step()
        if step%200==0:
            row={'step':step,'seconds':time.time()-start,'train':measure(model,train),'holdout':measure(model,validation)}
            arm['learning'].append(row);result['time']=time.time();atomic(out/'result.json',result)
            print(json.dumps({'arm':name,**row}),flush=True)
    if enabled:
        model.generator.head.neural_conjunction_gates=False;arm['same_weights_gates_disabled']=measure(model,validation)
        model.generator.head.neural_conjunction_gates=True
    weights={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith(('grounder.vision.','teacher.'))}
    path=out/f'{name}_diagnostic_weights.pt';torch.save(weights,path);arm['weights_sha256']=sha(path)
    arm['peak_cuda_gib']=torch.cuda.max_memory_allocated()/2**30
    del model,optimizer,weights;torch.cuda.empty_cache()
result.update(state='complete',time=time.time());atomic(out/'result.json',result)
print(json.dumps({'state':'complete','output':str(out)}),flush=True)
