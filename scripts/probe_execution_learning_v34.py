"""Matched collision-feedback study through generated anonymous discrete states."""
import collections,copy,json,os,random,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from PIL import Image
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,DataBank,image_tensor,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.symbols import tokenize,DiscreteState
from nesy_vla.outcome_supervision import SymbolicExecutionDecoder,execution_loss,rule_execution_cost

torch.set_num_threads(2);source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');cfg['recovery_demonstration_root']=str(base/'data/recovery_demonstrations_v27')
cfg['execution_outcome_root']='/home/user/xjy/strips_wm_navigation_20261009/data/search_cost_supervision_v3'
followon='--rule-cost' in sys.argv
setup(cfg);bank=DataBank(cfg);out=base/('rule_execution_learning_v35' if followon else 'execution_learning_v34');out.mkdir(exist_ok=True)
import fcntl
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (out/'result.json').exists():
    previous=read(out/'result.json')
    if previous['state']=='complete':raise RuntimeError('Completed probe must not be repeated')
    atomic(out/f'previous_attempt_{time.time_ns()}.json',previous)
result={'state':'extracting','pid':os.getpid(),'time':time.time(),'source_commit':read(source/'COMMITTED'),
    'provenance':bank.provenance,'original_development_or_fixed_used':False,
    'all_actions_via_generated_rules':True,'auxiliary_predictions_never_select_actions':True,
    'manual_predicate_meanings':False,'not_navigation_success':True,'arms':{}}
if followon:
    assert read(base/'execution_learning_v34/result.json')['state']=='complete'
    result['initialization']='Both arms continue the final600-step execution-feedback diagnostic; only measured rule cost differs.'
atomic(out/'result.json',result)
held=set(read(base/'conjunction_learning_v31/dataset.json')['probe_holdout_layouts']);selected=[]
for eid in sorted(bank.recovery_demo_ids):
    eligible=[r for r in bank.demos[eid] if 'execution_outcome_labels' in r]
    if not eligible:continue
    choices=sorted(set([0,len(eligible)//2]))
    for index in choices:
        row=eligible[index];ep=bank.by_id[eid]
        selected.append({'id':eid,'image':row['image'],'scene':ep['scene'],'category':ep['category'],
            'label':row['action'],'outcomes':row['execution_outcome_labels'],'heldout':ep['scene'] in held,'row':row})
initial=NavigationModel(cfg).eval();load_checkpoint(source,initial)
with torch.no_grad():
    for start in range(0,len(selected),8):
        batch=selected[start:start+8]
        images=[Image.open(bank.demo_image_path(e['row'])).convert('RGB') for e in batch]
        tokens=initial.grounder.encode(torch.cat([image_tensor(im,cfg['image_size'],'cpu') for im in images])).detach()
        for row,token in zip(batch,tokens):row['tokens']=token
del initial
train=[r for r in selected if not r['heldout']];validation=[r for r in selected if r['heldout']]
labels=torch.tensor([r['outcomes'] for r in train],dtype=torch.float32);rates=labels.mean((0,1))
positive=((1-rates)/rates.clamp_min(1e-5)).clamp(1,20)
atomic(out/'dataset.json',{'training_layouts':sorted({r['scene'] for r in train}),
    'holdout_training_layouts':sorted(held),'channel_training_positive_rates':rates.tolist(),
    'positive_weights':positive.tolist(),'examples':[{k:v for k,v in r.items() if k not in ['tokens','row']} for r in selected]})

def forward(model,rows,detach_aux=False):
    probabilities=[];actions=[];predicted=[];labels=[];codes=[]
    for category in ['chair','table']:
        chosen=[r for r in rows if r['category']==category]
        if not chosen:continue
        p=model.generator('Find the '+category+' and stop close to it.',[],[],None)
        tokens=torch.stack([r['tokens'] for r in chosen])
        h=model.grounder.query(p.definitions[None].expand(len(chosen),-1,-1),tokens)
        observation=model.grounder.truth(h).squeeze(-1).sigmoid();state=tokenize(observation)
        probabilities.append(model.logic.probabilities_for_states(p,state));codes.append(state.codes)
        auxiliary=DiscreteState(state.codes,state.values.detach()) if detach_aux else state
        predicted.append(model.execution_decoder(p.definitions.detach() if detach_aux else p.definitions,auxiliary))
        actions.extend(r['label'] for r in chosen);labels.extend(r['outcomes'] for r in chosen)
    return torch.cat(probabilities),torch.tensor(actions),torch.cat(predicted),torch.tensor(labels,dtype=torch.float32),torch.cat(codes)

@torch.no_grad()
def measure(model,rows):
    predictions=[];actions=[];outputs=[];labels=[];states=[];nll=0.
    for start in range(0,len(rows),32):
        p,y,logits,target,codes=forward(model,rows[start:start+32])
        predictions.extend(p.argmax(1).tolist());actions.extend(y.tolist());outputs.append(logits);labels.append(target);states.append(codes)
        nll+=float(-p.gather(1,y[:,None]).clamp_min(1e-9).log().sum())
    output=torch.cat(outputs);truth=torch.cat(labels).bool();est=output>0;summary={}
    for channel,name in enumerate(['collision','no_movement']):
        t=truth[:,:,channel];e=est[:,:,channel];tp=int((e&t).sum());tn=int((~e&~t).sum());pos=int(t.sum());neg=t.numel()-pos
        summary[name]={'positives':pos,'negatives':neg,'true_positives':tp,'true_negatives':tn,
            'balanced_accuracy':.5*(tp/max(1,pos)+tn/max(1,neg))}
    move=[(i,a) for i,a in enumerate(predictions) if a>0]
    return {'examples':len(rows),'expert_action_accuracy':sum(a==b for a,b in zip(predictions,actions))/len(rows),
        'turn_action_recall':sum(a==b and b not in [0,4,11] for a,b in zip(predictions,actions))/max(1,sum(b not in [0,4,11] for b in actions)),
        'greedy_counts':dict(collections.Counter(predictions)),'nll':nll/len(rows),
        'chosen_moves':len(move),'chosen_move_observed_collision_rate':sum(bool(truth[i,a-1,0]) for i,a in move)/max(1,len(move)),
        'chosen_move_no_movement_rate':sum(bool(truth[i,a-1,1]) for i,a in move)/max(1,len(move)),
        'auxiliary':summary,'unique_symbol_states':len(torch.unique(torch.cat(states),dim=0))}

result.update(state='training',frames=len(selected),train_frames=len(train),holdout_frames=len(validation),time=time.time())
atomic(out/'result.json',result)
arms=[('continued_aux_control',False),('measured_rule_cost',False)] if followon else [('bc_control',True),('execution_feedback',False)]
for name,detach_aux in arms:
    torch.manual_seed(3411);model=NavigationModel(cfg).eval();load_checkpoint(source,model)
    model.execution_decoder=SymbolicExecutionDecoder(cfg['program']['d'],cfg['program']['a'])
    if followon:
        weights=torch.load(base/'execution_learning_v34/execution_feedback_diagnostic_weights.pt',map_location='cpu',weights_only=True)
        mismatch=model.load_state_dict(weights,strict=False)
        assert not mismatch.unexpected_keys and all(k.startswith(('grounder.vision.','teacher.')) for k in mismatch.missing_keys)
    model.grounder.vision.requires_grad_(False);model.grounder.projection.requires_grad_(False);model.grounder.position.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=2e-4,weight_decay=.01)
    rng=random.Random(3411);arm={'initial':{'train':measure(model,train),'holdout':measure(model,validation)},'learning':[]}
    result['arms'][name]=arm;start=time.time()
    for step in range(1,(400 if followon else 600)+1):
        batch=rng.sample(train,min(32,len(train)));p,y,output,target,codes=forward(model,batch,detach_aux)
        bc=-p.gather(1,y[:,None]).clamp_min(1e-9).log().mean();aux=execution_loss(output,target,positive)
        if step==1:
            modules={'grounder':list(model.grounder.query.parameters())+list(model.grounder.truth.parameters()),
                     'generator':list(model.generator.parameters()),'auxiliary':list(model.execution_decoder.parameters())}
            arm['execution_gradient_norms']={}
            for module,values in modules.items():
                values=[v for v in values if v.requires_grad]
                grads=torch.autograd.grad(aux,values,retain_graph=True,allow_unused=True)
                arm['execution_gradient_norms'][module]=sum(float(g.detach().square().sum()) for g in grads if g is not None)**.5
        loss=bc+.5*aux
        if name=='measured_rule_cost':loss=loss+.5*rule_execution_cost(p,target)
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,2.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite discrete-outcome gradient')
        optimizer.step()
        if step%200==0:
            row={'step':step,'seconds':time.time()-start,'train':measure(model,train),'holdout':measure(model,validation)}
            arm['learning'].append(row);result['time']=time.time();atomic(out/'result.json',result)
            print(json.dumps({'arm':name,**row}),flush=True)
    torch.save({k:v for k,v in model.state_dict().items() if not k.startswith(('grounder.vision.','teacher.'))},out/f'{name}_diagnostic_weights.pt')
    del model,optimizer
result.update(state='complete',time=time.time());atomic(out/'result.json',result)
print(json.dumps({'state':'complete','output':str(out)}),flush=True)
