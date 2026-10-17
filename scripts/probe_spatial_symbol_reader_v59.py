"""Compare symbolic spatial readers on held-out ORIGINAL TRAIN layouts only."""
import fcntl,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'spatial_symbol_reader_v59';out.mkdir(exist_ok=True)
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));sys.path.insert(1,str(b/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.spatial_token_automaton import SpatialTokenAutomaton
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists(),'Completed probe cannot be repeated'
torch.set_num_threads(2)
cfg=read(b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
source=b/'visual_vocabulary_probe_v53';rows=read(source/'rows.json');manifest=read(source/'manifest.json')
assert not set(manifest['fit_scenes']+manifest['audit_training_scenes']).intersection(bank.provenance['development_layouts']+bank.provenance['fixed_layouts'])
codes=torch.tensor(np.load(source/'convnext_k64_tokens.npy').astype(np.int64),device='cuda')
assert codes.shape==(len(rows),49)
labels=torch.tensor([r['label'] for r in rows],device='cuda')
tasks=torch.tensor([['chair','table'].index(r['category']) for r in rows],device='cuda')
fit=torch.tensor([i for i,r in enumerate(rows) if r['probe_split']=='fit' and r['supervised']],device='cuda')
audit=torch.tensor([i for i,r in enumerate(rows) if r['probe_split']=='audit' and r['supervised']],device='cuda')
architecture=dict(patches=49,vocabulary=64,tasks=2,factors=4,states=16,actions=17,width=64,rank=4,reset_memory=False,task_banks=True,context_bridge=True)
torch.manual_seed(5900);initial=ProbabilisticTokenAutomaton(**architecture).state_dict()
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
def forward(model,index):
 x=codes[index];program=model.generate(tasks[index],x[:,None],0)
 _,prior=model.initial_state(len(index),'cuda');action=torch.full((len(index),),17,device='cuda',dtype=torch.long)
 return model.step(program,x,prior,action,0)
@torch.no_grad()
def measure(model,index):
 accuracy=0;loss=0.;symbols=torch.zeros(4,16,device='cuda');prediction=[]
 for start in range(0,len(index),256):
  ids=index[start:start+256];r=forward(model,ids);p=r['probabilities'];prediction.extend(p.argmax(-1).cpu().tolist())
  accuracy+=int((p.argmax(-1)==labels[ids]).sum());loss+=float(-p.clamp_min(1e-20).log().gather(1,labels[ids,None]).sum())
  symbols+=F.one_hot(r['observed_ids'],16).sum(0)
 return {'accuracy':accuracy/len(index),'nll':loss/len(index),'observation_ids_used':(symbols>0).sum(-1).tolist(),'predictions':prediction}
result={'started':time.time(),'source_manifest':manifest,'code_file_sha256':sha(source/'convnext_k64_tokens.npy'),'architecture':architecture,
 'qualification':'Single-frame action prediction using only integer tokens. Official heldouts excluded; no navigation result.',
 'fit_frames':len(fit),'audit_frames':len(audit),'updates':600,'batch_size':256,'arms':{}}
try:
 for arm in ['position_specific_mlp','shared_spatial']:
  torch.manual_seed(5900)
  if arm=='position_specific_mlp':model=ProbabilisticTokenAutomaton(**architecture).cuda();model.load_state_dict(initial)
  else:model=SpatialTokenAutomaton(**architecture).cuda();model.initialize_common_program(initial)
  model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.0001)
  generator=torch.Generator(device='cuda').manual_seed(5901);measures=[];started=time.time()
  for step in range(1,601):
   ids=fit[torch.randint(len(fit),(256,),device='cuda',generator=generator)]
   r=forward(model,ids);action=-r['probabilities'].clamp_min(1e-20).log().gather(1,labels[ids,None]).mean()
   recon=F.cross_entropy(r['reconstruction'].transpose(1,2),codes[ids])
   loss=action+.2*recon
   for distribution in [r['observed_probabilities'],r['state_probabilities']]:
    mean=distribution.mean(0);loss+=.005*(mean*(mean.clamp_min(1e-12)*16).log()).sum(-1).mean()
   optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step()
   if step%100==0:
    train=measure(model,fit);test=measure(model,audit)
    measures.append({'update':step,'fit':train,'audit':test,'seconds':time.time()-started})
    progress(state='training',arm=arm,update=step,fit_accuracy=train['accuracy'],audit_accuracy=test['accuracy'],audit_observation_ids_used=test['observation_ids_used'])
  result['arms'][arm]={'parameters':sum(p.numel() for p in model.parameters()),'measurements':measures}
  atomic(out/'progress.json',result);del model,optimizer;torch.cuda.empty_cache()
 result['finished']=time.time();atomic(out/'result.json',result);progress(state='complete',summary={k:{'fit_accuracy':v['measurements'][-1]['fit']['accuracy'],'audit_accuracy':v['measurements'][-1]['audit']['accuracy']} for k,v in result['arms'].items()})
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
