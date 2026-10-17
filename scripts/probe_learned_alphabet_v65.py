"""Task-trainable discrete visual alphabet; diagnostic on original TRAIN layouts."""
import fcntl,gc,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'learned_alphabet_v65';out.mkdir(exist_ok=True)
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));sys.path.insert(1,str(b/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank,image_tensor,load_checkpoint
from nesy_vla.model import NavigationModel
from nesy_vla.learned_visual_tokens import LearnedVisualAlphabet,TrainableAlphabetAutomaton
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists()
torch.set_num_threads(2)
cfg=read(b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
from stripsnav.io import committed_torch_save
source=b/'visual_vocabulary_probe_v53';rows=read(source/'rows.json');source_manifest=read(source/'manifest.json')
assert not set(source_manifest['fit_scenes']+source_manifest['audit_training_scenes']).intersection(bank.provenance['development_layouts']+bank.provenance['fixed_layouts'])
vocabulary=torch.load(source/'convnext_k64_vocabulary.pt',map_location='cuda',weights_only=True)
labels=torch.tensor([r['label'] for r in rows],device='cuda');tasks=torch.tensor([['chair','table'].index(r['category']) for r in rows],device='cuda')
fit=torch.tensor([i for i,r in enumerate(rows) if r['probe_split']=='fit' and r['supervised']],device='cuda')
audit=torch.tensor([i for i,r in enumerate(rows) if r['probe_split']=='audit' and r['supervised']],device='cuda')
architecture=dict(patches=49,vocabulary=64,tasks=2,factors=4,states=16,actions=17,width=64,rank=4,reset_memory=False,task_banks=True,context_bridge=True)
torch.manual_seed(5900);initial=TrainableAlphabetAutomaton(**architecture).state_dict()
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
manifest={'source_manifest':source_manifest,'rows_sha256':sha(source/'rows.json'),'vocabulary_sha256':sha(source/'convnext_k64_vocabulary.pt'),
 'architecture':architecture,'fit_frames':len(fit),'audit_frames':len(audit),'updates':600,'batch_size':256,
 'symbol_model_lr':1e-3,'alphabet_lr':1e-4,'alphabet_drift_weight':.005,'temperature':.2,
 'inputs_to_inference':'Only exact discrete tokens and anonymous finite-state beliefs. Pixel/PCA features stop at the quantizer.',
 'training_derivatives':'Straight-through categorical derivative at the visual alphabet. Forward computations equal deployment with ordinary integer IDs.',
 'qualification':'Single-frame action classification on held-out ORIGINAL TRAIN layouts, not a navigation result. Officialdev/fixed80 neither fitted nor evaluated.'}
atomic(out/'manifest.json',manifest)
result={'started':time.time(),'manifest':manifest,'arms':{}}
try:
 loc=b/'symbolic_conjunctive_v23/runs/conjunctive/best_development';vcfg=read(loc.parent/'config.json')
 visual=NavigationModel(vcfg).eval();load_checkpoint(loc,visual);visual.requires_grad_(False).cuda();encoder=visual.grounder.vision;del visual
 features=[]
 with torch.no_grad():
  for i,row in enumerate(rows):
   image=Image.open(row['path']).convert('RGB')
   with torch.autocast('cuda',dtype=torch.bfloat16):f=F.adaptive_avg_pool2d(encoder(image_tensor(image,224,'cuda')),(7,7)).flatten(2).transpose(1,2)
   features.append(f[0].float().cpu())
   if i%512==0:progress(state='encoding',images=i+1,total=len(rows))
  features=torch.stack(features).cuda();del encoder;gc.collect();torch.cuda.empty_cache()
  vectors=F.normalize(((features-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
  del features;gc.collect();torch.cuda.empty_cache()
 baseline=LearnedVisualAlphabet(vocabulary['centers']).cuda()
 with torch.no_grad():
  original_codes=torch.cat([baseline(vectors[i:i+128]).ids for i in range(0,len(rows),128)])
  stored=torch.tensor(np.load(source/'convnext_k64_tokens.npy').astype(np.int64),device='cuda')
  result['initial_code_agreement_with_saved_v53']=float((original_codes==stored).float().mean());del stored,baseline
 def forward(model,alphabet,index):
  tokens=alphabet(vectors[index]);program=model.generate(tasks[index],tokens[:,None],0)
  _,prior=model.initial_state(len(index),'cuda');action=torch.full((len(index),),17,device='cuda',dtype=torch.long)
  return model.step(program,tokens,prior,action,0),tokens
 @torch.no_grad()
 def measure(model,alphabet,index):
  correct=0;nll=0.;changed=0;total=0;ids_used=torch.zeros(64,device='cuda');prediction=[];max_error=0.
  for start in range(0,len(index),256):
   ids=index[start:start+256];r,tokens=forward(model,alphabet,ids);p=r['probabilities'];prediction.extend(p.argmax(-1).tolist())
   correct+=int((p.argmax(-1)==labels[ids]).sum());nll+=float(-p.clamp_min(1e-20).log().gather(1,labels[ids,None]).sum())
   changed+=int((tokens.ids!=original_codes[ids]).sum());total+=tokens.ids.numel();ids_used+=torch.bincount(tokens.ids.flatten(),minlength=64)
   # The entire generated program and action can be replayed after discarding
   # both images and all quantizer features/relaxations.
   program=model.generate(tasks[ids],tokens.ids[:,None],0);_,prior=model.initial_state(len(ids),'cuda');a=torch.full((len(ids),),17,device='cuda',dtype=torch.long)
   plain=model.step(program,tokens.ids,prior,a,0)
   max_error=max(max_error,float((plain['probabilities']-p).abs().max()));assert max_error==0
   proof=model.proof(program,plain['observed_ids'],prior,a,plain['state_values'],0);assert torch.equal(proof['probabilities'],p)
  return {'accuracy':correct/len(index),'nll':nll/len(index),'token_change_fraction':changed/total,
   'alphabet_ids_used':int((ids_used>0).sum()),'integer_only_replay_max_error':max_error,'predictions':prediction}
 for arm in ['frozen_alphabet','learned_alphabet']:
  torch.manual_seed(5900);model=TrainableAlphabetAutomaton(**architecture).cuda();model.load_state_dict(initial)
  alphabet=LearnedVisualAlphabet(vocabulary['centers']).cuda();alphabet.requires_grad_(arm=='learned_alphabet')
  params=[{'params':list(model.parameters()),'lr':1e-3}]
  if arm=='learned_alphabet':params.append({'params':list(alphabet.parameters()),'lr':1e-4})
  optimizer=torch.optim.AdamW(params,weight_decay=.0001);generator=torch.Generator(device='cuda').manual_seed(5901)
  measures=[];best=-1.;started=time.time()
  for step in range(1,601):
   ids=fit[torch.randint(len(fit),(256,),device='cuda',generator=generator)]
   r,tokens=forward(model,alphabet,ids)
   loss=-r['probabilities'].clamp_min(1e-20).log().gather(1,labels[ids,None]).mean()
   loss=loss+.2*F.cross_entropy(r['reconstruction'].transpose(1,2),tokens.ids)+.005*alphabet.drift_penalty()
   for distribution in [r['observed_probabilities'],r['state_probabilities']]:
    mean=distribution.mean(0);loss+=.005*(mean*(mean.clamp_min(1e-12)*16).log()).sum(-1).mean()
   optimizer.zero_grad(set_to_none=True);loss.backward()
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
   qnorm=torch.nn.utils.clip_grad_norm_(alphabet.parameters(),1.);assert torch.isfinite(qnorm)
   optimizer.step()
   if step%100==0:
    train=measure(model,alphabet,fit);test=measure(model,alphabet,audit)
    measures.append({'update':step,'fit':train,'audit':test,'seconds':time.time()-started,'alphabet_gradient_norm':float(qnorm)})
    if test['accuracy']>best:
     best=test['accuracy'];committed_torch_save(out/arm/'best_audit',{'model':model.state_dict(),'alphabet':alphabet.state_dict(),'update':step,'manifest':manifest},
      {'epoch':step,'optimizer_included':False,'scope':'component audit only; not a navigation checkpoint'})
    progress(state='training',arm=arm,update=step,fit_accuracy=train['accuracy'],audit_accuracy=test['accuracy'],
     alphabet_gradient_norm=float(qnorm),audit_token_change_fraction=test['token_change_fraction'],integer_only_replay_max_error=test['integer_only_replay_max_error'])
  result['arms'][arm]={'measurements':measures,'best_audit_accuracy':best};atomic(out/'progress.json',result)
  del model,alphabet,optimizer;gc.collect();torch.cuda.empty_cache()
 result['finished']=time.time();atomic(out/'result.json',result);progress(state='complete',summary={a:r['best_audit_accuracy'] for a,r in result['arms'].items()})
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
