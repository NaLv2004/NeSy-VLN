"""Automatically discover spatial token predicates; induce executable rules.

Images stop at vector quantization. Learned Boolean clauses are the sole path
from integer token identities to action probabilities. This is a movement-only
capacity probe using original training scenes, never a navigation evaluation.
"""
import collections,fcntl,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12'
sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,DataBank,image_tensor,atomic,sha
from nesy_vla.model import NavigationModel

torch.set_num_threads(2);torch.manual_seed(4200);device='cuda:0'
out=base/'spatial_symbol_rules_v42';out.mkdir(exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists(),'Do not duplicate an existing probe'
dataset=read(base/'execution_learning_v34/dataset.json')
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');cfg['recovery_demonstration_root']=str(base/'data/recovery_demonstrations_v27')
setup(cfg);bank=DataBank(cfg)
lookup={(eid,r['image']):r for eid in bank.recovery_demo_ids for r in bank.demos[eid]}
model=NavigationModel(cfg).eval();load_checkpoint(source,model);model.requires_grad_(False).to(device)
rows=dataset['examples'];train_index=np.array([i for i,r in enumerate(rows) if not r['heldout']])
held_index=np.array([i for i,r in enumerate(rows) if r['heldout']])
assert len(train_index)==609 and len(held_index)==155
result={'state':'encoding','time':time.time(),'pid':os.getpid(),'source_commit':read(source/'COMMITTED'),
        'data_sha256':sha(base/'execution_learning_v34/dataset.json'),'fixed_and_development_used':False,
        'predicate_semantics':'Unsupervised learned visual cluster membership at image-patch addresses; no semantic predicate labels',
        'action_path':'Integer tokens -> automatically induced true/false conjunctions -> learned leaf conclusions -> rule vote',
        'not_navigation_success':True,'arms':{}}
atomic(out/'result.json',result)
features=[]
with torch.no_grad():
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]
        images=[Image.open(bank.demo_image_path(lookup[(r['id'],r['image'])])).convert('RGB') for r in batch]
        inp=torch.cat([image_tensor(im,cfg['image_size'],device) for im in images])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=model.grounder.vision(inp)
            f=torch.nn.functional.adaptive_avg_pool2d(f,(7,7)).flatten(2).transpose(1,2)
        features.append(f.float())
features=torch.cat(features);del model;torch.cuda.empty_cache()
training=features[train_index].flatten(0,1)
mean=training.mean(0);sd=training.std(0).clamp_min(.01)
z=(training-mean)/sd
eigenvalues,axes=torch.linalg.eigh(z.T@z/len(z))
axes=axes[:,-128:]
vectors=((features-mean)/sd)@axes
vectors=torch.nn.functional.normalize(vectors,dim=-1)
training=vectors[train_index].flatten(0,1)
for clusters in [64,256]:
    torch.manual_seed(4200)
    centers=training[torch.randperm(len(training),device=device)[:clusters]].clone()
    for iteration in range(30):
        dist=torch.cdist(training,centers);assignment=dist.argmin(1)
        counts=torch.bincount(assignment,minlength=clusters)
        new=torch.zeros_like(centers).index_add_(0,assignment,training)
        alive=counts>0;centers[alive]=new[alive]/counts[alive,None]
    codes=torch.cdist(vectors.flatten(0,1),centers).argmin(1).reshape(len(rows),49).cpu().numpy()
    assert np.all(codes>=0) and np.all(codes<clusters)
    # Cluster equality predicates are discovered from training images; no code
    # ordering, geometric labels, target locations or object masks are supplied.
    col=(np.arange(49)[None]*clusters+codes).ravel()
    rr=np.repeat(np.arange(len(rows)),49)
    X=csr_matrix((np.ones(len(col)),(rr,col)),shape=(len(rows),49*clusters))
    atomic(out/f'tokens_{clusters}.json',{'examples':[dict(r,codes=c.tolist()) for r,c in zip(rows,codes)],
                                        'codebook_training_examples':609,'heldout_used_for_codebook':False})
    torch.save({'mean':mean.cpu(),'sd':sd.cpu(),'axes':axes.cpu(),'centers':centers.cpu()},out/f'codebook_{clusters}.pt')
    for kind,klass in [('forest',RandomForestClassifier),('extra',ExtraTreesClassifier)]:
        name=f'{kind}_k{clusters}';forests={};probabilities=np.zeros((len(rows),17))
        for category in ['chair','table']:
            chosen=np.array([i for i in train_index if rows[i]['category']==category])
            learner=klass(n_estimators=64,max_depth=18,min_samples_leaf=2,max_features=.3,n_jobs=2,random_state=4200)
            learner.fit(X[chosen],np.array([rows[i]['label'] for i in chosen]))
            clauses=[]
            for estimator in learner.estimators_:
                rules=[];tree=estimator.tree_
                def visit(node,positive,negative):
                    feature=int(tree.feature[node])
                    if feature<0:
                        value=tree.value[node,0].copy();value/=value.sum()
                        conclusion=np.zeros(17);conclusion[learner.classes_.astype(int)]=value
                        rules.append({'true':positive,'false':negative,'conclusion':conclusion.tolist()});return
                    assert 0<=tree.threshold[node]<1
                    literal=[feature//clusters,feature%clusters]
                    visit(tree.children_left[node],positive,negative+[literal])
                    visit(tree.children_right[node],positive+[literal],negative)
                visit(0,[],[]);clauses.append(rules)
            forests[category]=clauses
            indices=np.array([i for i,r in enumerate(rows) if r['category']==category])
            local=codes[indices];predicted=np.zeros((len(indices),17))
            for rules in clauses:
                fired=np.zeros(len(indices),dtype=int)
                for rule in rules:
                    active=np.ones(len(indices),dtype=bool)
                    for patch,token in rule['true']:active &= local[:,patch]==token
                    for patch,token in rule['false']:active &= local[:,patch]!=token
                    fired+=active;predicted+=active[:,None]*np.array(rule['conclusion'])[None]
                assert (fired==1).all()
            predicted/=len(clauses)
            expected=np.zeros_like(predicted);expected[:,learner.classes_.astype(int)]=learner.predict_proba(X[indices])
            assert np.allclose(predicted,expected,atol=1e-12)
            probabilities[indices]=predicted
        metrics={}
        for split,indices in [('train',train_index),('holdout',held_index)]:
            p=probabilities[indices];choice=p.argmax(1);labels=np.array([rows[i]['label'] for i in indices])
            outcomes=np.array([rows[i]['outcomes'] for i in indices]);moving=choice>0;turns=~np.isin(labels,[0,4,11])
            metrics[split]={'count':len(indices),'expert_correct':int((choice==labels).sum()),
                            'expert_accuracy':float((choice==labels).mean()),
                            'turn_correct':int(((choice==labels)&turns).sum()),'turn_count':int(turns.sum()),
                            'collisions':int(outcomes[np.arange(len(indices))[moving],choice[moving]-1,0].sum()),
                            'counts':dict(collections.Counter(map(int,choice)))}
        atomic(out/(name+'_rules.json'),forests)
        result['arms'][name]=metrics;result.update(state='rule_learning',time=time.time());atomic(out/'result.json',result)
        print(json.dumps({'arm':name,**metrics}),flush=True)
result.update(state='complete',time=time.time());atomic(out/'result.json',result)
