"""Matched discrete-description study on isolated ORIGINAL TRAIN layouts only."""
import argparse,fcntl,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from safetensors import safe_open
B=Path('/home/user/xjy/nesy_vla_20261010')
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(1,str(B/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.semantic_token_automaton import SemanticTokens,SemanticTokenAutomaton

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['visual_only','aligned_text','shuffled_text'],required=True)
 args=p.parse_args();out=B/'semantic_component_v69'/args.arm;out.mkdir(parents=True,exist_ok=True)
 lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (out/'result.json').exists(),'Completed study cannot be repeated'
 state={'pid':os.getpid(),'state':'initializing','time':time.time()}
 def progress(**kw):
  state.update(kw,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
 try:
  torch.set_num_threads(2)
  cfg=read(B/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
  from stripsnav.io import committed_torch_save
  corpus=B/'semantic_tokens_v68/corpus';source=B/'visual_vocabulary_probe_v53'
  rows=read(source/'rows.json');provenance=read(source/'manifest.json')
  assert not set(provenance['fit_scenes']+provenance['audit_training_scenes']).intersection(bank.provenance['development_layouts']+bank.provenance['fixed_layouts'])
  captions={};identities=set();ledgers={}
  for rank in range(8):
   manifest=read(corpus/f'rank{rank}.manifest.json');done=read(corpus/f'rank{rank}.result.json')
   assert done['completed']==done['total']==len(manifest['targets'])
   identities.add(manifest['frontend_sha256']);ledgers[str(rank)]=sha(corpus/f'rank{rank}.jsonl')
   for line in (corpus/f'rank{rank}.jsonl').read_text().splitlines():
    item=json.loads(line);assert item['frontend_sha256']==manifest['frontend_sha256'];h=item['image_sha256']
    assert h not in captions;captions[h]=item
  assert len(identities)==1
  texts=[]
  for row in rows:
   assert bank.by_id[row['episode_id']]['scene'] in bank.provenance['train_layouts']
   texts.append(captions[sha(row['path'])]['token_ids'])
  assert all(0<len(x)<=96 for x in texts)
  words=torch.zeros((len(rows),96),dtype=torch.long,device='cuda');mask=torch.zeros_like(words,dtype=torch.bool)
  for i,t in enumerate(texts):words[i,:len(t)]=torch.tensor(t,device='cuda');mask[i,:len(t)]=True
  codes=torch.tensor(np.load(source/'convnext_k64_tokens.npy').astype(np.int64),device='cuda')
  labels=torch.tensor([r['label'] for r in rows],device='cuda');tasks=torch.tensor([['chair','table'].index(r['category']) for r in rows],device='cuda')
  fit=torch.tensor([i for i,r in enumerate(rows) if r['probe_split']=='fit' and r['supervised']],device='cuda')
  audit=torch.tensor([i for i,r in enumerate(rows) if r['probe_split']=='audit' and r['supervised']],device='cuda')
  permutation=torch.arange(len(rows),device='cuda');rng=torch.Generator(device='cuda').manual_seed(6902)
  for partition in (fit,audit):permutation[partition]=partition[torch.randperm(len(partition),device='cuda',generator=rng)]
  # Shuffling stays inside each fit/audit split; no audit text can enter training.
  assert set(permutation[fit].tolist())==set(fit.tolist())
  frontend=manifest['frontend'];model_path=Path(frontend['model']);index=read(model_path/'model.safetensors.index.json')
  key='model.language_model.embed_tokens.weight';weight_file=model_path/index['weight_map'][key]
  assert sha(weight_file)==frontend['weight_sha256'][weight_file.name]
  with safe_open(str(weight_file),framework='pt',device='cpu') as handle:table=handle.get_tensor(key)
  architecture=dict(patches=49,vocabulary=64,tasks=2,factors=4,states=16,actions=17,width=64,rank=4,reset_memory=False,task_banks=True,context_bridge=True)
  torch.manual_seed(5900);base=ProbabilisticTokenAutomaton(**architecture)
  model=SemanticTokenAutomaton(table,**architecture);model.initialize_common(base.state_dict());del base,table
  model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.0001)
  generator=torch.Generator(device='cuda').manual_seed(5901)
  experiment={'arm':args.arm,'architecture':architecture,'rows_sha256':sha(source/'rows.json'),'vocabulary_sha256':sha(source/'convnext_k64_vocabulary.pt'),
   'frontend':frontend,'frontend_sha256':next(iter(identities)),'caption_ledgers_sha256':ledgers,'fit_frames':len(fit),'audit_frames':len(audit),
   'updates':600,'batch_size':256,'seed':5900,'word_table_key':key,'word_table_source_sha256':frontend['weight_sha256'][weight_file.name],
   'qualification':'Single-frame component accuracy on held-out ORIGINAL TRAIN layouts. This is not navigation success. Official development/fixed80 not used.',
   'constraint':'Frozen VLM emits discrete words; only discrete word IDs and visual patch IDs enter learned predicates/programs. All action probabilities from finite-state transitions and soft rules.'}
  atomic(out/'manifest.json',experiment)
  def tokens(ids,mode=None):
   mode=mode or args.arm;idx=permutation[ids] if mode=='shuffled_text' else ids
   return SemanticTokens(codes[ids],words[idx],torch.zeros_like(mask[idx]) if mode=='visual_only' else mask[idx])
  def forward(ids,mode=None):
   x=tokens(ids,mode);program=model.generate(tasks[ids],x[:,None],0)
   _,prior=model.initial_state(len(ids),'cuda');action=torch.full((len(ids),),17,dtype=torch.long,device='cuda')
   return model.step(program,x,prior,action,0),program,prior,action
  @torch.no_grad()
  def measure(ids,mode=None):
   correct=0;nll=0.;error=0.;predictions=[];usage=[set() for _ in range(4)]
   for start in range(0,len(ids),256):
    batch=ids[start:start+256];r,program,prior,action=forward(batch,mode);p=r['probabilities']
    correct+=int((p.argmax(-1)==labels[batch]).sum());nll-=float(p.clamp_min(1e-20).log().gather(1,labels[batch,None]).sum())
    proof=model.proof(program,r['observed_ids'],prior,action,r['state_values'],0)
    error=max(error,float((proof['probabilities']-p).abs().max()));assert error==0
    for f in range(4):usage[f].update(r['observed_ids'][:,f].tolist())
    predictions.extend(p.argmax(-1).tolist())
   return {'accuracy':correct/len(ids),'nll':nll/len(ids),'symbolic_proof_max_error':error,'observed_ids_used':list(map(len,usage)),'predictions':predictions}
  result={'manifest':experiment,'started':time.time(),'measurements':[]};best=-1.
  for update in range(1,601):
   ids=fit[torch.randint(len(fit),(256,),device='cuda',generator=generator)];r,_,_,_=forward(ids)
   loss=-r['probabilities'].clamp_min(1e-20).log().gather(1,labels[ids,None]).mean()
   loss+=.2*F.cross_entropy(r['reconstruction'].transpose(1,2),codes[ids])
   for distribution in [r['observed_probabilities'],r['state_probabilities']]:
    average=distribution.mean(0);loss+=.005*(average*(average.clamp_min(1e-12)*16).log()).sum(-1).mean()
   optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
   optimizer.step()
   if update%100==0:
    train=measure(fit);test=measure(audit)
    evidence={'update':update,'fit':train,'audit':test,'seconds':time.time()-result['started'],'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    if args.arm=='aligned_text':
     evidence['audit_text_erased']=measure(audit,'visual_only');evidence['audit_text_shuffled']=measure(audit,'shuffled_text')
    result['measurements'].append(evidence)
    if test['accuracy']>best:
     best=test['accuracy'];committed_torch_save(out/'best_audit',{'model':model.state_dict(),'update':update,'manifest':experiment},
      {'epoch':update,'optimizer_included':False,'scope':'Training-layout component audit, not a navigation checkpoint'})
    atomic(out/'progress.json',result);progress(state='training',update=update,fit_accuracy=train['accuracy'],audit_accuracy=test['accuracy'],
     symbolic_proof_max_error=test['symbolic_proof_max_error'],peak_gpu_allocated_gib=evidence['peak_gpu_allocated_gib'])
  result.update(finished=time.time(),best_audit_accuracy=best);atomic(out/'result.json',result);progress(state='complete',best_audit_accuracy=best)
 except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
