"""Compare automatically learned patch alphabets on held-out TRAIN layouts.

The official development/fixed80 layouts are not sampled or fitted. This is a
component diagnostic: categorical-rule action prediction, not navigation.
"""
import fcntl,gc,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import csr_matrix
from sklearn.ensemble import ExtraTreesClassifier
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'visual_vocabulary_probe_v53';out.mkdir(exist_ok=True)
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));sys.path.insert(1,str(b/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank,load_checkpoint,image_tensor
from nesy_vla.model import NavigationModel
from nesy_vla.vision_backbone import FrozenQwenVision
torch.set_num_threads(2);torch.manual_seed(5300)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists(),'Completed probe cannot be repeated'
cfg=read(b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
source=b/'spatial_automaton_v43';all_rows=read(source/'training_rows.json')
assert read(source/'COMMITTED')['data_provenance']==bank.provenance
train_scenes=bank.provenance['train_layouts'];audit_scenes=set(train_scenes[-4:]);fit_scenes=set(train_scenes)-audit_scenes
assert not (fit_scenes|audit_scenes).intersection(bank.provenance['development_layouts']+bank.provenance['fixed_layouts'])
grouped={}
for i,r in enumerate(all_rows):grouped.setdefault(r['episode_id'],[]).append(i)
rng=random.Random(5300);chosen=[];selected_episodes={}
for name,scenes,target_rows in [('fit',fit_scenes,4096),('audit',audit_scenes,1024)]:
 candidates=[eid for eid in grouped if bank.by_id[eid]['scene'] in scenes];rng.shuffle(candidates);n=0;selected_episodes[name]=[]
 for eid in candidates:
  selected_episodes[name].append(eid)
  for i in grouped[eid]:chosen.append((i,name))
  n+=len(grouped[eid])
  if n>=target_rows:break
 assert n>=target_rows,(name,n)
rows=[dict(all_rows[i],probe_split=split,source_index=i) for i,split in chosen]
train_index=np.array([i for i,r in enumerate(rows) if r['probe_split']=='fit' and r['supervised']])
audit_index=np.array([i for i,r in enumerate(rows) if r['probe_split']=='audit' and r['supervised']])
for r in rows:assert Path(r['path']).is_file(),r['path']
manifest={'time':time.time(),'source_rows_sha256':sha(source/'training_rows.json'),
 'fit_scenes':sorted(fit_scenes),'audit_training_scenes':sorted(audit_scenes),'selected_episodes':selected_episodes,
 'rows':len(rows),'fit_supervised_rows':len(train_index),'audit_supervised_rows':len(audit_index),
 'official_development_or_fixed80_used':False,'fit_quantizer_on_audit_images':False,
 'image_size':224,'canonical_encoder_batch':1,'predicate_semantics':'Automatically discovered cluster equality at49 patch positions; anonymous learned IDs.',
 'action_path':'Discrete current and previous tokens -> induced Boolean rule conjunctions -> soft rule conclusions.',
 'qualification':'Action accuracy on held-out training layouts is not navigation success.'}
atomic(out/'manifest.json',manifest);atomic(out/'rows.json',rows)
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
images=[Image.open(r['path']).convert('RGB') for r in rows]
result={'manifest':manifest,'arms':{}}
try:
 for arm in ['convnext','qwen_vision']:
  started=time.time();torch.manual_seed(5300)
  if arm=='convnext':
   loc=b/'symbolic_conjunctive_v23/runs/conjunctive/best_development';scfg=read(loc.parent/'config.json');visual=NavigationModel(scfg).eval();load_checkpoint(loc,visual);visual.requires_grad_(False).cuda();encoder=visual.grounder.vision;del visual
  else:encoder=FrozenQwenVision(cfg['model_path'],microbatch=1).cuda()
  features=[]
  with torch.no_grad():
   for i,image in enumerate(images):
    with torch.autocast('cuda',dtype=torch.bfloat16):f=F.adaptive_avg_pool2d(encoder(image_tensor(image,224,'cuda')),(7,7)).flatten(2).transpose(1,2)
    features.append(f[0].float().cpu())
    if i%256==0:progress(state='encoding',arm=arm,images=i+1,total=len(images))
  features=torch.stack(features).cuda();del encoder;gc.collect();torch.cuda.empty_cache()
  # Full empirical covariance uses only training images and uniformly spaced
  # patch samples. No task/action/geometry labels shape the visual alphabet.
  fitting=features[train_index,::3].flatten(0,1);mean=fitting.mean(0);sd=fitting.std(0).clamp_min(.01)
  z=(fitting-mean)/sd;_,axes=torch.linalg.eigh(z.T@z/len(z));axes=axes[:,-128:]
  vectors=F.normalize(((features-mean)/sd)@axes,dim=-1)
  fit_vectors=vectors[train_index,::3].flatten(0,1);del features,fitting,z;torch.cuda.empty_cache()
  for clusters in [64,256]:
   torch.manual_seed(5300);centers=fit_vectors[torch.randperm(len(fit_vectors),device='cuda')[:clusters]].clone()
   for iteration in range(40):
    assignment=torch.cdist(fit_vectors,centers).argmin(1);counts=torch.bincount(assignment,minlength=clusters);sums=torch.zeros_like(centers).index_add_(0,assignment,fit_vectors);alive=counts>0;centers[alive]=sums[alive]/counts[alive,None]
   codes=[]
   for start in range(0,len(vectors),128):codes.append(torch.cdist(vectors[start:start+128].flatten(0,1),centers).argmin(1).reshape(-1,49).cpu().numpy())
   codes=np.concatenate(codes);context=np.full((len(rows),98),clusters,dtype=np.int64);context[:,:49]=codes
   for i in range(1,len(rows)):
    if rows[i]['episode_id']==rows[i-1]['episode_id']:context[i,49:]=codes[i-1]
   cols=(np.arange(98)[None]*(clusters+1)+context).ravel();X=csr_matrix((np.ones(len(cols)),(np.repeat(np.arange(len(rows)),98),cols)),shape=(len(rows),98*(clusters+1)))
   labels=np.array([r['label'] for r in rows]);prediction=np.zeros(len(rows),dtype=int);by_category={}
   for category in ['chair','table']:
    fit=np.array([i for i in train_index if rows[i]['category']==category]);test=np.array([i for i in audit_index if rows[i]['category']==category]);forest=ExtraTreesClassifier(n_estimators=96,max_depth=24,min_samples_leaf=2,max_features=.3,n_jobs=2,random_state=5300)
    progress(state='fitting_symbolic_rules',arm=arm,clusters=clusters,category=category)
    forest.fit(X[fit],labels[fit]);prediction[test]=forest.predict(X[test]);by_category[category]={'fit':len(fit),'audit':len(test),'fit_accuracy':float((forest.predict(X[fit])==labels[fit]).mean()),'audit_accuracy':float((prediction[test]==labels[test]).mean())};del forest
   name=f'{arm}_k{clusters}';freq=np.bincount(codes[train_index].ravel(),minlength=clusters)
   m={'training_codes_used':int((freq>0).sum()),'audit_action_accuracy':float((prediction[audit_index]==labels[audit_index]).mean()),'by_category':by_category,'seconds_since_encoder_start':time.time()-started,
      'audit_action_predictions':prediction[audit_index].tolist(),'audit_indices':audit_index.tolist()}
   result['arms'][name]=m
   torch.save({'mean':mean.cpu(),'sd':sd.cpu(),'axes':axes.cpu(),'centers':centers.cpu()},out/f'{name}_vocabulary.pt')
   np.save(out/f'{name}_tokens.npy',codes.astype(np.int16));atomic(out/'progress.json',result)
   del X,context,centers;gc.collect()
  del vectors,fit_vectors,mean,sd,axes;gc.collect();torch.cuda.empty_cache()
 atomic(out/'result.json',dict(result,finished=time.time()));progress(state='complete',summary={k:{x:y for x,y in v.items() if x not in ['audit_action_predictions','audit_indices']} for k,v in result['arms'].items()})
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
