"""Matched train-only learning probe with fixed visual features and full rules."""
import collections,copy,json,os,random,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from PIL import Image
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,DataBank,image_tensor,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.symbols import tokenize
torch.set_num_threads(2);source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');setup(cfg);bank=DataBank(cfg)
out=base/'conjunction_learning_v31';out.mkdir(exist_ok=False)
sampler=random.Random(3100);groups={}
for eid in bank.demo_ids:
    episode=bank.by_id[eid];groups.setdefault((episode['scene'],episode['category']),[]).append(eid)
selected=[]
for key,ids in sorted(groups.items()):
    sampler.shuffle(ids);selected.extend(ids[:8])
scenes=sorted({bank.by_id[e]['scene'] for e in selected});held=set(scenes[::5])
initial=NavigationModel(cfg).eval();load_checkpoint(source,initial);examples=[]
with torch.no_grad():
    for start in range(0,len(selected),8):
        ids=selected[start:start+8];images=[Image.open(bank.demo_image_path(bank.demos[e][0])).convert('RGB') for e in ids]
        tokens=initial.grounder.encode(torch.cat([image_tensor(im,cfg['image_size'],'cpu') for im in images])).detach()
        for eid,token in zip(ids,tokens):
            episode=bank.by_id[eid];examples.append({'id':eid,'scene':episode['scene'],'category':episode['category'],
                'label':bank.demos[eid][0]['action'],'tokens':token,'heldout':episode['scene'] in held})
train=[e for e in examples if not e['heldout']];validation=[e for e in examples if e['heldout']]
atomic(out/'dataset.json',{'source_commit':read(source/'COMMITTED'),'train_only':True,'original_fixed_or_dev_used':False,
    'probe_training_layouts':sorted(set(scenes)-held),'probe_holdout_layouts':sorted(held),
    'examples':[{k:v for k,v in e.items() if k!='tokens'} for e in examples]})
del initial
def forward(model,rows):
    probabilities=[];labels=[]
    for category in ['chair','table']:
        chosen=[r for r in rows if r['category']==category]
        if not chosen:continue
        program=model.generator('Find the '+category+' and stop close to it.',[],[],None)
        tokens=torch.stack([r['tokens'] for r in chosen])
        h=model.grounder.query(program.definitions[None].expand(len(chosen),-1,-1),tokens)
        observation=model.grounder.truth(h).squeeze(-1).sigmoid()
        probabilities.append(model.logic.probabilities_for_states(program,tokenize(observation)))
        labels.extend(r['label'] for r in chosen)
    return torch.cat(probabilities),torch.tensor(labels)
@torch.no_grad()
def measure(model,rows):
    predictions=[];labels=[];nll=0.
    for start in range(0,len(rows),32):
        p,y=forward(model,rows[start:start+32]);predictions.extend(p.argmax(1).tolist());labels.extend(y.tolist())
        nll+=float(-p.gather(1,y[:,None]).clamp_min(1e-9).log().sum())
    counts=collections.Counter(labels);correct=collections.Counter(y for p,y in zip(predictions,labels) if p==y)
    return {'examples':len(rows),'accuracy':sum(a==b for a,b in zip(predictions,labels))/len(rows),
        'macro_recall':sum(correct[a]/counts[a] for a in counts)/len(counts),'nll':nll/len(rows),
        'greedy_counts':dict(collections.Counter(predictions)),
        'turn_action_recall':sum(a==b and b not in [0,4,11] for a,b in zip(predictions,labels))/max(1,sum(y not in [0,4,11] for y in labels))}
result={'state':'running','pid':os.getpid(),'time':time.time(),'source_commit':read(source/'COMMITTED'),
    'all_actions_via_discrete_rules':True,'cached_visual_features_only_inside_grounder':True,
    'no_manual_predicate_labels':True,'not_navigation_success':True,'arms':{}}
atomic(out/'result.json',result)
for operator in ['product','minimum']:
    torch.manual_seed(3111);local=copy.deepcopy(cfg);local['program']['conjunction_operator']=operator
    model=NavigationModel(local).eval();load_checkpoint(source,model)
    model.grounder.vision.requires_grad_(False);model.grounder.projection.requires_grad_(False)
    model.grounder.position.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=2e-4,weight_decay=.01);rng=random.Random(3111)
    arm={'initial':{'train':measure(model,train),'heldout':measure(model,validation)},'learning':[]}
    result['arms'][operator]=arm;start=time.time()
    for step in range(1,401):
        batch=rng.sample(train,min(32,len(train)));p,y=forward(model,batch)
        loss=-p.gather(1,y[:,None]).clamp_min(1e-9).log().mean()
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,2.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite rule/grounding gradient')
        optimizer.step()
        if step%100==0:
            row={'step':step,'seconds':time.time()-start,'train':measure(model,train),'heldout':measure(model,validation)}
            arm['learning'].append(row);result['time']=time.time();atomic(out/'result.json',result)
            print(json.dumps({'operator':operator,**row}),flush=True)
    torch.save({k:v for k,v in model.state_dict().items() if not k.startswith(('grounder.vision.','teacher.'))},out/f'{operator}_diagnostic_weights.pt')
    del model,optimizer
result.update(state='complete',time=time.time());atomic(out/'result.json',result)
print(json.dumps({'state':'complete','out':str(out)}),flush=True)
