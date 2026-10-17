"""Stratify completed train-only probes by privately audited visibility."""
import copy,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from PIL import Image
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,DataBank,image_tensor,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.symbols import tokenize,tokenize_categories
torch.set_num_threads(2);source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');setup(cfg);bank=DataBank(cfg)
visibility=read(base/'visibility_replay_v33/status.json');assert visibility['state']=='complete'
annotations={r['id']:r for r in visibility['rows']};assert all(r['rgb_identical'] for r in annotations.values())
examples=read(base/'conjunction_learning_v31/dataset.json')['examples'];initial=NavigationModel(cfg).eval();load_checkpoint(source,initial)
with torch.no_grad():
    for start in range(0,len(examples),8):
        batch=examples[start:start+8];images=[Image.open(bank.demo_image_path(bank.demos[e['id']][0])).convert('RGB') for e in batch]
        tokens=initial.grounder.encode(torch.cat([image_tensor(im,cfg['image_size'],'cpu') for im in images]))
        for row,token in zip(batch,tokens):row['tokens']=token;row['visible']=annotations[row['id']]['visible_pixels']>=16
del initial
result={}
with torch.no_grad():
    for name,operator,groups,weights in [('initial','product',0,None),
        ('product400','product',0,base/'conjunction_learning_v31/product_diagnostic_weights.pt'),
        ('minimum400','minimum',0,base/'conjunction_learning_v31/minimum_diagnostic_weights.pt'),
        ('categorical400','product',8,base/'categorical_learning_v32/product_diagnostic_weights.pt')]:
        local=copy.deepcopy(cfg);local['program']['conjunction_operator']=operator
        model=NavigationModel(local).eval();load_checkpoint(source,model)
        if weights:
            missing=model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=False)
            assert not missing.unexpected_keys and all(k.startswith(('grounder.vision.','teacher.')) for k in missing.missing_keys)
        predicted=[]
        for category in ['chair','table']:
            p=model.generator('Find the '+category+' and stop close to it.',[],[],None)
            chosen=[r for r in examples if r['category']==category]
            for start in range(0,len(chosen),32):
                batch=chosen[start:start+32];tokens=torch.stack([r['tokens'] for r in batch])
                h=model.grounder.query(p.definitions[None].expand(len(batch),-1,-1),tokens)
                logits=model.grounder.truth(h).squeeze(-1)
                state=tokenize_categories(logits,groups) if groups else tokenize(logits.sigmoid())
                probs=model.logic.probabilities_for_states(p,state)
                for r,a in zip(batch,probs.argmax(1).tolist()):
                    predicted.append({'id':r['id'],'heldout':r['heldout'],'visible':r['visible'],'label':r['label'],'prediction':a})
        summary={}
        for split,held in [('train',False),('probe_holdout',True)]:
            for category,visible in [('visible',True),('not_visible',False),('all',None)]:
                rows=[r for r in predicted if r['heldout']==held and (visible is None or r['visible']==visible)]
                turn=[r for r in rows if r['label'] not in [0,4,11]]
                summary[split+'/'+category]={'frames':len(rows),'correct':sum(r['label']==r['prediction'] for r in rows),
                    'accuracy':sum(r['label']==r['prediction'] for r in rows)/len(rows),
                    'turn_examples':len(turn),'turn_correct':sum(r['label']==r['prediction'] for r in turn)}
        result[name]={'summary':summary,'examples':predicted};del model
atomic(base/'visibility_probe_breakdown_v33.json',{'time':time.time(),'train_only':True,'not_navigation_success':True,
    'source_commit':read(source/'COMMITTED'),'results':result})
print(json.dumps({name:r['summary'] for name,r in result.items()},indent=2))
