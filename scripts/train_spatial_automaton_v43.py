"""Full-data exploratory discrete-token automaton; original development only.

All rule predicates (patch token equal/not equal) and rule conclusions are
induced from training data. No RGB feature or continuous visual value enters
the interpreter. This alternative studies representation capacity; it does not
replace the existing differentiable program generator/revision experiment.
"""
import collections,fcntl,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.sparse import csr_matrix
from sklearn.ensemble import ExtraTreesClassifier

base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,image_tensor,atomic,sha,pil
from nesy_vla.model import NavigationModel
from nesy_vla.evaluation import aggregate

torch.set_num_threads(2);torch.manual_seed(4300);device='cuda:0';K=64
out=base/'spatial_automaton_v43';out.mkdir(exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'status.json').exists(),'Inspect the original handle before resuming; do not duplicate'
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(base/'symbolic_execution_v38/configs/grounded_outcome_feedback.json')
source_cfg=read(source.parent/'config.json');setup(cfg);bank=DataBank(cfg)
state={'state':'initializing','pid':os.getpid(),'time':time.time()};atomic(out/'status.json',state)
def progress(**fields):
    state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
manifest={'source_commit':read(source/'COMMITTED'),'data':bank.provenance,
          'horizon':32,'sampling':'greedy','original_fixed80_used':False,
          'development_ids':[e['episode_id'] for e in bank.dev],
          'task_symbols':sorted({bank.by_id[e]['category'] for e in bank.demo_ids}),
          'quantizer':'Training-only 128D PCA and 64 learned visual prototypes at 7x7 patch addresses',
          'memory':'Previous discrete patch tokens and previous action token; initialized with an unseen token',
          'rules':'96 induced decision trees per training task token. Every path is an AND of learned equality/inequality predicates. Leaf conclusions are learned distributions, combined by equal soft vote.',
          'limitations':'Exploratory alternative with fixed induced rule programs; no online ten-step program revision. Not a replacement for the main learned revision architecture.'}
atomic(out/'manifest.json',manifest)
model=NavigationModel(source_cfg).eval();load_checkpoint(source,model);model.requires_grad_(False).to(device)
vision=model.grounder.vision
records=[];episode_ranges={}
for eid in bank.demo_ids:
    begin=len(records)
    for row in bank.demos[eid]:
        records.append({'episode_id':eid,'category':bank.by_id[eid]['category'],'path':str(bank.demo_image_path(row)),
                        'label':row['action'],'supervised':row.get('supervision_mask',True)})
    episode_ranges[eid]=(begin,len(records))
assert all(type(r['supervised']) is bool for r in records)
labels=np.array([r['label'] for r in records]);supervised=np.array([r['supervised'] for r in records])
atomic(out/'training_rows.json',records)

@torch.no_grad()
def encode(images):
    tensors=torch.cat([image_tensor(im,source_cfg['image_size'],device) for im in images])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        features=vision(tensors)
        return torch.nn.functional.adaptive_avg_pool2d(features,(7,7)).flatten(2).transpose(1,2).float()

@torch.no_grad()
def codes_for(images):
    f=encode(images);vectors=torch.nn.functional.normalize(((f-mean)/sd)@axes,dim=-1)
    return torch.cdist(vectors.flatten(0,1),centers).argmin(1).reshape(len(images),49).cpu().numpy().astype(np.int16)

sim=None
try:
    rng=np.random.default_rng(4300);selected=rng.choice(np.flatnonzero(supervised),min(2048,int(supervised.sum())),replace=False)
    feature_parts=[]
    for start in range(0,len(selected),32):
        images=[Image.open(records[i]['path']).convert('RGB') for i in selected[start:start+32]]
        # Uniform patch subsampling reduces fitting cost; all chosen images are training only.
        f=encode(images);feature_parts.append(f[:,::3].reshape(-1,f.shape[-1]))
        if start==0:progress(state='fitting_visual_vocabulary',encoded_codebook_images=len(images),training_rows=len(records))
    features=torch.cat(feature_parts);mean=features.mean(0);sd=features.std(0).clamp_min(.01)
    z=(features-mean)/sd;_,axes=torch.linalg.eigh(z.T@z/len(z));axes=axes[:,-128:]
    vectors=torch.nn.functional.normalize(z@axes,dim=-1)
    centers=vectors[torch.randperm(len(vectors),device=device)[:K]].clone()
    for iteration in range(40):
        assignment=torch.cdist(vectors,centers).argmin(1);counts=torch.bincount(assignment,minlength=K)
        sums=torch.zeros_like(centers).index_add_(0,assignment,vectors);alive=counts>0
        centers[alive]=sums[alive]/counts[alive,None]
    torch.save({'mean':mean.cpu(),'sd':sd.cpu(),'axes':axes.cpu(),'centers':centers.cpu()},out/'visual_vocabulary.pt')
    del features,feature_parts,z,vectors;torch.cuda.empty_cache()
    all_codes=np.empty((len(records),49),dtype=np.int16)
    for start in range(0,len(records),32):
        images=[Image.open(r['path']).convert('RGB') for r in records[start:start+32]]
        all_codes[start:start+len(images)]=codes_for(images)
        if start%1024==0:progress(state='encoding_training_images',encoded=min(start+32,len(records)),training_rows=len(records))
    np.save(out/'visual_tokens.npy',all_codes)
    context=np.full((len(records),99),K,dtype=np.int16);context[:,:49]=all_codes
    for begin,end in episode_ranges.values():
        context[begin+1:end,49:98]=all_codes[begin:end-1]
        context[begin+1:end,98]=labels[begin:end-1]
    def sparse(values):
        columns=(np.arange(values.shape[1])[None]*(K+1)+values).ravel()
        return csr_matrix((np.ones(len(columns)),(np.repeat(np.arange(len(values)),values.shape[1]),columns)),shape=(len(values),99*(K+1)))
    X=sparse(context);programs={};training_metrics={};reference_predictions={}
    for category in manifest['task_symbols']:
        indices=np.array([i for i,r in enumerate(records) if r['category']==category and r['supervised']])
        progress(state='inducing_rules',task_symbol=category,supervised_rows=len(indices))
        learner=ExtraTreesClassifier(n_estimators=96,max_depth=24,min_samples_leaf=2,max_features=.3,n_jobs=2,random_state=4300)
        learner.fit(X[indices],labels[indices]);trees=[]
        for estimator in learner.estimators_:
            tree=estimator.tree_;values=tree.value[:,0].copy();values/=values.sum(1,keepdims=True)
            conclusions=np.zeros((tree.node_count,17));conclusions[:,learner.classes_.astype(int)]=values
            assert np.all((tree.threshold[tree.feature>=0]>=0)&(tree.threshold[tree.feature>=0]<1))
            trees.append({'predicate':tree.feature.tolist(),'false_branch':tree.children_left.tolist(),
                          'true_branch':tree.children_right.tolist(),
                          'conclusion':[c.tolist() if tree.feature[j]<0 else None for j,c in enumerate(conclusions)]})
        programs[category]=trees
        reference=np.zeros((min(100,len(indices)),17))
        reference[:,learner.classes_.astype(int)]=learner.predict_proba(X[indices[:100]])
        reference_predictions[category]=reference
        training_metrics[category]={'n':len(indices),'expert_accuracy':float((learner.predict(X[indices])==labels[indices]).mean()),
                                    'trees':len(trees),'nodes':sum(len(t['predicate']) for t in trees)}
    atomic(out/'programs.json',programs,compact=True);atomic(out/'training_metrics.json',training_metrics)

    def infer(category,integer_state):
        assert integer_state.shape==(99,) and integer_state.dtype.kind in 'iu'
        assert np.all((integer_state>=0)&(integer_state<=K))
        total=np.zeros(17);proof=[]
        for tree_index,tree in enumerate(programs[category]):
            node=0;clauses=[]
            while tree['predicate'][node]>=0:
                predicate=tree['predicate'][node];slot,token=divmod(predicate,K+1)
                truth=bool(integer_state[slot]==token);clauses.append([slot,token,truth])
                node=tree['true_branch' if truth else 'false_branch'][node]
            total+=tree['conclusion'][node]
            proof.append({'tree':tree_index,'leaf':node,'clauses':clauses})
        probabilities=total/len(programs[category])
        return probabilities,proof

    # Validate serialized symbolic execution on actual training states against the
    # induced learners independently, before any simulator receives an action.
    for category in manifest['task_symbols']:
        indices=[i for i,r in enumerate(records) if r['category']==category and r['supervised']][:100]
        trees=programs[category]
        for offset,i in enumerate(indices):
            p,proof=infer(category,context[i]); reconstructed=np.zeros(17)
            assert np.allclose(p,reference_predictions[category][offset],atol=1e-12)
            for certificate in proof:
                assert all((context[i,slot]==token)==truth for slot,token,truth in certificate['clauses'])
                reconstructed+=trees[certificate['tree']]['conclusion'][certificate['leaf']]
            assert np.allclose(p,reconstructed/len(trees))
    atomic(out/'COMMITTED',{'program_sha256':sha(out/'programs.json'),'vocabulary_sha256':sha(out/'visual_vocabulary.pt'),
                           'data_provenance':bank.provenance,'time':time.time(),'optimizer_included':False})
    del X,context;torch.cuda.empty_cache()
    sim=bank.simulator(out/'simulator.log',0,evaluation=True)
    atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pid':sim.process.pid,'time':time.time()})
    evaluated=[]
    for episode in bank.dev:
        folder=out/'development'/episode['episode_id'];folder.mkdir(parents=True,exist_ok=True)
        response=sim.call('reset',episode=episode);image=pil(response['rgb']);image.save(folder/'first.png')
        previous=np.full(49,K,dtype=np.int16);previous_action=K;trace=[]
        for step in range(32):
            codes=codes_for([image])[0];symbol_state=np.concatenate([codes,previous,np.array([previous_action],dtype=np.int16)])
            p,proof=infer(episode['category'],symbol_state);action=int(p.argmax())
            frame_sha=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            response=sim.call('step',action=action)
            trace.append({'step':step,'symbols':symbol_state.tolist(),'action':action,'probabilities':p.tolist(),
                          'proof':proof,'image_sha256':frame_sha,'collision':bool(response.get('feedback',{}).get('collision',False))})
            image=pil(response['rgb']);previous=codes;previous_action=action
            if response['done']:break
        image.save(folder/'last.png');metrics=response.get('metrics',{})
        row={'episode_id':episode['episode_id'],'scene':episode['scene'],'domain':episode['domain'],
             'category':episode['category'],'distance_group':episode['distance_group'],
             'success':bool(metrics.get('success',False)),'steps':len(trace),'collisions':sum(t['collision'] for t in trace)}
        atomic(folder/'trace.json',{'episode':episode,'metrics':metrics,'steps':trace},compact=True)
        atomic(folder/'evaluation.json',row);evaluated.append(row)
        progress(state='evaluating_development',summary=aggregate(evaluated))
    atomic(out/'development/summary.json',aggregate(evaluated));progress(state='complete',summary=aggregate(evaluated))
except BaseException as error:
    progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    if sim is not None:sim.close()
