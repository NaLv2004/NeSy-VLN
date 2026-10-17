"""Train a visual encoder THROUGH discrete tokens, never around symbolic rules."""
import argparse,fcntl,gc,json,os,shutil,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
B=Path('/home/user/xjy/nesy_vla_20261010')
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(1,str(B/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank,image_tensor,load_checkpoint
from nesy_vla.model import NavigationModel
from nesy_vla.learned_visual_tokens import LearnedVisualAlphabet,TrainableAlphabetAutomaton

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['frozen','tail_1e5','tail_1e4'],required=True);args=p.parse_args()
 out=B/'perceptual_learning_v72'/args.arm;out.mkdir(parents=True,exist_ok=True)
 lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (out/'result.json').exists(),'Completed probe cannot be repeated'
 state={'pid':os.getpid(),'state':'initializing','time':time.time()}
 def progress(**kw):
  state.update(kw,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
 try:
  torch.set_num_threads(2);cfg=read(B/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
  source=B/'visual_vocabulary_probe_v53';rows=read(source/'rows.json');source_manifest=read(source/'manifest.json')
  assert not set(source_manifest['fit_scenes']+source_manifest['audit_training_scenes']).intersection(bank.provenance['development_layouts']+bank.provenance['fixed_layouts'])
  vocabulary=torch.load(source/'convnext_k64_vocabulary.pt',map_location='cuda',weights_only=True)
  labels=torch.tensor([r['label'] for r in rows],device='cuda');tasks=torch.tensor([['chair','table'].index(r['category']) for r in rows],device='cuda')
  fit=torch.tensor([i for i,r in enumerate(rows) if r['probe_split']=='fit' and r['supervised']],device='cuda')
  audit=torch.tensor([i for i,r in enumerate(rows) if r['probe_split']=='audit' and r['supervised']],device='cuda')
  loc=B/'symbolic_conjunctive_v23/runs/conjunctive/best_development';vcfg=read(loc.parent/'config.json')
  visual=NavigationModel(vcfg).eval();load_checkpoint(loc,visual);visual.requires_grad_(False).cuda();encoder=visual.grounder.vision;del visual
  if hasattr(encoder,'backbone'):encoder=encoder.backbone
  assert isinstance(encoder,torch.nn.Sequential) and len(encoder)==8,'Unexpected encoder architecture'
  prefix=encoder[:6].eval().requires_grad_(False);tail=encoder[6:].eval();del encoder
  features=[]
  with torch.no_grad():
   for offset in range(0,len(rows),32):
    images=torch.cat([image_tensor(Image.open(r['path']).convert('RGB'),224,'cuda') for r in rows[offset:offset+32]])
    with torch.autocast('cuda',dtype=torch.bfloat16):encoded=prefix(images)
    features.append(encoded.cpu())
    if offset%1024==0:progress(state='frozen_prefix',completed=min(offset+32,len(rows)),total=len(rows))
   features=torch.cat(features).cuda()
  del prefix;gc.collect();torch.cuda.empty_cache()
  lr={'frozen':0.,'tail_1e5':1e-5,'tail_1e4':1e-4}[args.arm];tail.requires_grad_(lr>0)
  alphabet=LearnedVisualAlphabet(vocabulary['centers']).cuda().requires_grad_(False)
  architecture=dict(patches=49,vocabulary=64,tasks=2,factors=4,states=16,actions=17,width=64,rank=4,reset_memory=False,task_banks=True,context_bridge=True)
  torch.manual_seed(5900);model=TrainableAlphabetAutomaton(**architecture).cuda()
  groups=[{'params':model.parameters(),'lr':1e-3}]
  if lr:groups.append({'params':tail.parameters(),'lr':lr})
  optimizer=torch.optim.AdamW(groups,weight_decay=.0001);rng=torch.Generator(device='cuda').manual_seed(5901)
  original=torch.tensor(np.load(source/'convnext_k64_tokens.npy').astype(np.int64),device='cuda')
  manifest={'arm':args.arm,'architecture':architecture,'rows_sha256':sha(source/'rows.json'),'vocabulary_sha256':sha(source/'convnext_k64_vocabulary.pt'),
   'prefix_source':str(loc),'prefix_source_commit':read(loc/'COMMITTED'),'vision_lr':lr,'model_lr':1e-3,'batch_size':256,'updates':600,
   'fit_frames':len(fit),'audit_frames':len(audit),'source_split':source_manifest,'trainable_visual_parameters':sum(p.numel() for p in tail.parameters() if p.requires_grad),
   'constraint':'Visual values stop at a hard integer patch alphabet. Deployed integer-only replay and every action rule proof must be exact. Only gradients cross the categorical boundary.',
   'qualification':'Original-training component probe only. No development/fixed80 data, no navigation claim. Frozen prefix cached in RAM only.'}
  atomic(out/'manifest.json',manifest)
  def forward(ids):
   with torch.autocast('cuda',dtype=torch.bfloat16):f=F.adaptive_avg_pool2d(tail(features[ids]),(7,7)).flatten(2).transpose(1,2)
   vectors=F.normalize(((f.float()-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
   tokens=alphabet(vectors);program=model.generate(tasks[ids],tokens[:,None],0)
   _,prior=model.initial_state(len(ids),'cuda');a=torch.full((len(ids),),17,dtype=torch.long,device='cuda')
   return model.step(program,tokens,prior,a,0),tokens,prior,a
  @torch.no_grad()
  def measure(ids):
   correct=0;error=0.;changed=0;total=0;predictions=[];usage=[set() for _ in range(4)]
   for offset in range(0,len(ids),256):
    batch=ids[offset:offset+256];r,tokens,prior,a=forward(batch);prob=r['probabilities']
    correct+=int((prob.argmax(-1)==labels[batch]).sum());predictions.extend(prob.argmax(-1).tolist())
    program=model.generate(tasks[batch],tokens.ids[:,None],0);plain=model.step(program,tokens.ids,prior,a,0)
    error=max(error,float((plain['probabilities']-prob).abs().max()));assert error==0
    proof=model.proof(program,plain['observed_ids'],prior,a,plain['state_values'],0);assert torch.equal(proof['probabilities'],prob)
    changed+=int((tokens.ids!=original[batch]).sum());total+=tokens.ids.numel()
    for factor in range(4):usage[factor].update(r['observed_ids'][:,factor].tolist())
   return {'accuracy':correct/len(ids),'integer_only_replay_max_error':error,'token_change_fraction':changed/total,'observed_ids_used':list(map(len,usage)),'predictions':predictions}
  result={'manifest':manifest,'started':time.time(),'measurements':[]};best=-1.;best_payload=None
  for update in range(1,601):
   ids=fit[torch.randint(len(fit),(256,),device='cuda',generator=rng)];r,tokens,_,_=forward(ids)
   loss=-r['probabilities'].clamp_min(1e-20).log().gather(1,labels[ids,None]).mean()
   loss+=.2*F.cross_entropy(r['reconstruction'].transpose(1,2),tokens.ids)
   for distribution in [r['observed_probabilities'],r['state_probabilities']]:
    mean=distribution.mean(0);loss+=.005*(mean*(mean.clamp_min(1e-12)*16).log()).sum(-1).mean()
   optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
   visual_norm=torch.nn.utils.clip_grad_norm_(tail.parameters(),1.) if lr else torch.tensor(0.)
   assert torch.isfinite(visual_norm)
   if lr and update==1:assert visual_norm>0,'Visual encoder received no gradient through discrete rules'
   optimizer.step()
   if update%100==0:
    train=measure(fit);test=measure(audit)
    record={'update':update,'fit':train,'audit':test,'seconds':time.time()-result['started'],'visual_gradient_norm':float(visual_norm),
     'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    result['measurements'].append(record)
    if test['accuracy']>best:
     best=test['accuracy'];best_payload={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
      'visual_tail':{k:v.detach().cpu().clone() for k,v in tail.state_dict().items()} if lr else None,'epoch':update,'manifest':manifest}
    atomic(out/'progress.json',result);progress(state='training',update=update,fit_accuracy=train['accuracy'],audit_accuracy=test['accuracy'],
     visual_gradient_norm=float(visual_norm),peak_gpu_allocated_gib=record['peak_gpu_allocated_gib'],integer_only_replay_max_error=test['integer_only_replay_max_error'])
  # This standalone component has one bounded-size final artifact, no rolling
  # multi-gigabyte checkpoints. Retain a 512MiB reserve plus 3x tensor bytes.
  estimated=sum(t.numel()*t.element_size() for group in [best_payload['model'],best_payload['visual_tail'] or {}] for t in group.values())
  assert shutil.disk_usage(out).free>512*2**20+3*estimated,'Insufficient space for atomic component save'
  folder=out/'best_audit';folder.mkdir(exist_ok=True);path=folder/'checkpoint.pt';temp=folder/'checkpoint.partial.pt'
  with temp.open('wb') as f:torch.save(best_payload,f);f.flush();os.fsync(f.fileno())
  os.replace(temp,path);atomic(folder/'COMMITTED',{'checkpoint_file':path.name,'checkpoint_sha256':sha(path),'epoch':best_payload['epoch'],
   'optimizer_included':False,'scope':'Training-only component probe; frozen visual prefix referenced by immutable source commit'})
  result.update(finished=time.time(),best_audit_accuracy=best);atomic(out/'result.json',result);progress(state='complete',best_audit_accuracy=best)
 except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
