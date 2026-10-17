"""Encode full successful training sequences for the next generated-FSM study."""
import argparse,fcntl,gc,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
B=Path('/home/user/xjy/nesy_vla_20261010');sys.path.insert(0,str(B/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank,image_tensor,load_checkpoint
from nesy_vla.model import NavigationModel
from nesy_vla.learned_visual_tokens import LearnedVisualAlphabet

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['frozen','tail_1e4'],required=True);args=p.parse_args()
 out=B/'perceptual_sequences_v73'/args.arm;out.mkdir(parents=True,exist_ok=True)
 lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (out/'TOKENS_COMMITTED').exists(),'Complete sequences cannot be repeated'
 state={'pid':os.getpid(),'state':'loading','time':time.time()}
 def progress(**kw):
  state.update(kw,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
 try:
  torch.set_num_threads(2);cfg=read(B/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
  from stripsnav.io import load_committed
  primary=B/'spatial_automaton_v43/training_rows.json';recovery=B/'data/onpolicy_recovery_v48/training_rows.json'
  rows=read(primary);original_count=len(rows);rows+=read(recovery)
  train_scenes=set(bank.provenance['train_layouts']);groups={}
  for row in rows:
   source_id=row.get('source_episode_id',row['episode_id'])
   assert bank.by_id[source_id]['scene'] in train_scenes
   groups.setdefault(row['episode_id'],[]).append(row)
  assert all(len(group)<=32 and group[-1]['label']==0 for group in groups.values())
  atomic(out/'training_rows.json',rows)
  loc=B/'symbolic_conjunctive_v23/runs/conjunctive/best_development';visual=NavigationModel(read(loc.parent/'config.json')).eval()
  load_checkpoint(loc,visual);visual.requires_grad_(False).cuda();encoder=visual.grounder.vision;del visual
  if hasattr(encoder,'backbone'):encoder=encoder.backbone
  assert isinstance(encoder,torch.nn.Sequential) and len(encoder)==8
  warm=B/'perceptual_learning_v72'/args.arm/'best_audit';weights=load_committed(warm)
  if args.arm!='frozen':encoder[6:].load_state_dict(weights['visual_tail'])
  vocab_path=B/'visual_vocabulary_probe_v53/convnext_k64_vocabulary.pt';vocabulary=torch.load(vocab_path,map_location='cuda',weights_only=True)
  alphabet=LearnedVisualAlphabet(vocabulary['centers']).cuda().requires_grad_(False)
  manifest={'arm':args.arm,'data_provenance':bank.provenance,'source_rows_sha256':sha(primary),'recovery_rows_sha256':sha(recovery),
   'vocabulary_path':str(vocab_path),'vocabulary_sha256':sha(vocab_path),'encoder_source':str(loc),'encoder_source_commit':read(loc/'COMMITTED'),
   'component_checkpoint':str(warm),'component_commit':read(warm/'COMMITTED'),'canonical_visual_batch':1,
   'old_count':original_count,'rows':len(rows),'episodes':len(groups),'horizon':32,'revision_interval':10,
   'scope':'Training only. Integer codes only saved; no maps or visual feature cache. Same full-sequence supervision as v56.'}
  atomic(out/'manifest.json',manifest);del weights;gc.collect();torch.cuda.empty_cache()
  codes=np.empty((len(rows),49),dtype=np.uint8);started=time.time()
  with torch.no_grad():
   for i,row in enumerate(rows):
    with Image.open(row['path']) as image:x=image_tensor(image.convert('RGB'),224,'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):f=F.adaptive_avg_pool2d(encoder(x),(7,7)).flatten(2).transpose(1,2)
    vectors=F.normalize(((f.float()-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
    tokens=alphabet(vectors);codes[i]=tokens.ids[0].cpu().numpy().astype(np.uint8)
    if (i+1)%1024==0:progress(state='encoding',completed=i+1,total=len(rows),seconds=time.time()-started)
  target=out/'visual_tokens.npy';temp=out/'visual_tokens.partial.npy';np.save(temp,codes);os.replace(temp,target)
  record=dict(manifest,tokens_sha256=sha(target),training_rows_sha256=sha(out/'training_rows.json'),time=time.time(),
   ids_used=len(np.unique(codes)),seconds=time.time()-started)
  atomic(out/'TOKENS_COMMITTED',record);progress(state='complete',completed=len(rows),total=len(rows),seconds=record['seconds'])
 except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
