"""One bounded train-only state-refinement experiment; main PPO stays live."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import fcntl,hashlib,json,random,sys,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
B=Path('/home/user/xjy/nesy_vla_20261010');OUT=B/'memory_consequence_v97'
sys.path.insert(0,str(OUT/'src'));sys.path.insert(1,str(B/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.consequence_refinement import refine_automaton,ConsequenceDecoder,balanced_consequence_loss,merge_child_belief,discrete_memory_equalities
from nesy_vla.observation_inverse import observation_inverse_logits
from nesy_vla.outcome_supervision import rule_execution_cost

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
torch.manual_seed(9301);random.seed(9301);rng=random.Random(9301)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'COMPLETE').exists(),'Do not repeat completed refinement'
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**kw):state.update(kw,time=time.time());atomic(OUT/'status.json',state);print(json.dumps(state),flush=True)
try:
 cfg=read(B/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
 from stripsnav.io import load_committed
 source=B/'perceptual_sequences_v73/tail_1e4';commit=read(source/'TOKENS_COMMITTED')
 assert bank.provenance==commit['data_provenance']
 assert sha(source/'training_rows.json')==commit['training_rows_sha256']
 assert sha(source/'visual_tokens.npy')==commit['tokens_sha256']
 rows=read(source/'training_rows.json');codes=np.load(source/'visual_tokens.npy')
 grouped=defaultdict(list)
 for i,r in enumerate(rows):
  eid=r.get('source_episode_id',r['episode_id']);assert bank.by_id[eid]['scene'] in bank.provenance['train_layouts']
  grouped[r['episode_id']].append(i)
 episodes=[]
 for eid,indices in grouped.items():
  r=rows[indices[0]];sid=r.get('source_episode_id',eid);scene=bank.by_id[sid]['scene']
  assert rows[indices[-1]]['label']==0 and len(indices)<=32
  episodes.append(dict(id=eid,scene=scene,indices=indices,task=['chair','table'].index(r['category'])))
 outcome_path=Path(bank.provenance['execution_outcomes']['path'])
 assert sha(outcome_path)==bank.provenance['execution_outcomes']['sha256']
 records=read(outcome_path);index={r['path']:i for i,r in enumerate(rows)}
 outcome=np.zeros((len(rows),16,2),np.float32);has=np.zeros(len(rows),bool)
 for rec in records:
  path=str(Path('/home/user/xjy/strips_wm_navigation_20261009/data')/rec['image'])
  assert path in index,path
  i=index[path];assert rows[i]['episode_id']==rec['episode_id'] and rows[i]['supervised'] and rows[i]['label']!=0
  assert not has[i];has[i]=True
  outcome[i]=np.asarray([rec['observed_collision'][1:],[not v for v in rec['execution_valid'][1:]]],np.float32).T
 assert int(has.sum())==3355
 # These are two ORIGINAL TRAIN layouts, held out only from this refinement's
 # optimizer. Parent SFT/PPO has seen them; this is not an independent benchmark.
 probe_layouts={'layout_118','layout_119'}
 train=[e for e in episodes if e['scene'] not in probe_layouts]
 probe=[e for e in episodes if e['scene'] in probe_layouts]
 train_risk=[e for e in train if has[e['indices']].any()]
 probe_risk=[e for e in probe if has[e['indices']].any()]
 assert train_risk and probe_risk
 warm=B/'consequence_refinement_v93';checkpoint_file=warm/'checkpoint_0400.pt'
 assert sha(checkpoint_file)=='36e80f961febe8d8c9f4d5f7a4b94f4f2a6199ffd8360309bc6a25a35b77e4e4'
 assert read(warm/'COMPLETE')['completed_updates']==400
 saved=torch.load(checkpoint_file,map_location='cpu',weights_only=False)
 warm_commit=dict(checkpoint_file=checkpoint_file.name,checkpoint_sha256=sha(checkpoint_file),epoch=400)
 architecture=saved['manifest']['architecture'];assert architecture['states']==32
 model=ProbabilisticTokenAutomaton(**architecture).cuda();model.load_state_dict(saved['model'])
 decoder=ConsequenceDecoder(model.factors,model.states).cuda();decoder.load_state_dict(saved['decoder'])
 memory_decoder=ConsequenceDecoder(model.factors,model.states).cuda()
 source_optimizer=saved['optimizer'];source_model={k:v.detach().clone() for k,v in model.state_dict().items()}
 manifest=dict(source_checkpoint=warm_commit,source_path=str(warm),architecture=architecture,
   data=bank.provenance,source_tokens=commit,train_episodes=len(train),probe_episodes=len(probe),
   train_risk_episodes=len(train_risk),probe_risk_episodes=len(probe_risk),probe_layouts=sorted(probe_layouts),
   measured_frames=int(has.sum()),updates=400,batch_episodes=16,actor_lr=3e-5,
   consequence_weight=.5,memory_consequence_weight=.5,rule_execution_cost_weight=.2,head_warmup_batches=400,
   frozen_visual_frontend=True,source_checkpoint_navigation_repeated=False,
   scope='One bounded refinement preflight. No dev/fixed80 navigation. Probe layouts belong to original training and were seen by parent.',
   action_path='Integer49 patch IDs -> automatically learned hard observation IDs -> explicit finite-state belief transition -> generated conjunction rules -> action.',
   initialization='Resume exact v93 actor, observation decoder and actor optimizer. New training-only decoder receives hard anonymous memory IDs; its gradients break identical sibling transition conclusions.',
   teacher_signals='Existing measured outcomes supervise both hard observed equalities and hard memory equalities. Both auxiliary decoders discarded at deployment. No manually assigned state meanings or runtime action masks.')
 atomic(OUT/'manifest.json',manifest);del saved
 all_codes=torch.tensor(codes.astype(np.int64),device='cuda');labels=torch.tensor([r['label'] for r in rows],device='cuda')
 supervision=torch.tensor([r['supervised'] for r in rows],device='cuda');targets=torch.tensor(outcome,device='cuda');has_tensor=torch.tensor(has,device='cuda')
 weights=torch.tensor([4. if i>=commit['old_count'] else 1. for i in range(len(rows))],device='cuda')

 def batch_tensors(batch):
  length=max(len(e['indices']) for e in batch);ix=torch.zeros(len(batch),length,device='cuda',dtype=torch.long);valid=torch.zeros_like(ix,dtype=torch.bool)
  for j,e in enumerate(batch):
   k=len(e['indices']);ix[j,:k]=torch.tensor(e['indices'],device='cuda');valid[j,:k]=True
  return ix,valid,torch.tensor([e['task'] for e in batch],device='cuda')

 initialization=dict(source_sha256=sha(checkpoint_file),source_update=400,state_count=32,
  verification='Strict source actor and decoder loading; memory head does not participate in action selection.')
 atomic(OUT/'initialization_verification.json',initialization)

 def unroll(batch,training=False,capture=False):
  ix,valid,task=batch_tensors(batch);x=all_codes[ix];y=labels[ix];mask=valid&supervision[ix]
  weight=weights[ix]*mask;denom=valid.sum().clamp_min(1)
  _,belief=model.initial_state(len(batch),'cuda');last=torch.full((len(batch),),17,device='cuda',dtype=torch.long)
  total=x.new_zeros((),dtype=torch.float);captured=[];logits_all=[];outcomes=[];ids=[];tasks_all=[];probs=[];actions=[];expected=[];risk_probs=[];risk_actions=[];memory_logits=[];memory_ids=[];memory_captured=[]
  risk_count=(valid&has_tensor[ix]).sum().clamp_min(1)
  for step in range(ix.shape[1]):
   if step%10==0:program=model.generate(task,x[:,max(0,step-10):step+1],step)
   before=belief;r=model.step(program,x[:,step],belief,last,step)
   active=valid[:,step];imitation=-r['probabilities'].clamp_min(1e-20).log().gather(1,y[:,step,None]).squeeze(1)
   total=total+(imitation*weight[:,step]).sum()/weight.sum().clamp_min(1)
   if training:
    reconstruct=F.cross_entropy(r['reconstruction'].transpose(1,2),x[:,step],reduction='none').mean(-1)
    total=total+.2*(reconstruct*active).sum()/denom
    if step+1<ix.shape[1]:
     prediction=model.next_observation(r['state_values'],y[:,step])
     prediction_loss=F.cross_entropy(prediction.transpose(1,2),x[:,step+1],reduction='none').mean(-1)
     total=total+.2*(prediction_loss*valid[:,step+1]).sum()/denom
    if step:
     inverse=observation_inverse_logits(model,program,x[:,step-1],x[:,step],step)
     total=total+.1*(F.cross_entropy(inverse,y[:,step-1],reduction='none')*active).sum()/denom
   m=active&has_tensor[ix[:,step]]
   if m.any():
    hard=r['observed_values'][m];prediction=decoder(hard,task[m]);target=targets[ix[:,step][m]]
    hard_memory=discrete_memory_equalities(r['state_values'][m]);memory_prediction=memory_decoder(hard_memory,task[m])
    memory_logits.append(memory_prediction.detach());memory_ids.append(r['state_ids'][m].detach())
    if training:
     total=total+.5*balanced_consequence_loss(prediction,target)*m.sum()/risk_count
     total=total+.5*balanced_consequence_loss(memory_prediction,target)*m.sum()/risk_count
     total=total+.2*rule_execution_cost(r['probabilities'][m],target)*m.sum()/risk_count
    logits_all.append(prediction.detach());outcomes.append(target.detach());ids.append(r['observed_ids'][m].detach())
    if capture:captured.append(hard.detach());memory_captured.append(hard_memory.detach())
    tasks_all.append(task[m].detach())
    risk_probs.append(r['probabilities'][m].detach());risk_actions.append(y[:,step][m].detach())
   if mask[:,step].any():
    probs.append(r['probabilities'][mask[:,step]].detach());actions.append(y[:,step][mask[:,step]].detach())
   if not training:
    proof=model.proof(program,r['observed_ids'],before,last,r['state_values'],step)
    assert torch.equal(proof['probabilities'],r['probabilities'])
   belief=r['state_values'];last=y[:,step]
  return total,dict(logits=logits_all,outcomes=outcomes,ids=ids,captured=captured,tasks=tasks_all,probabilities=probs,actions=actions,risk_probabilities=risk_probs,memory_logits=memory_logits,memory_ids=memory_ids,memory_captured=memory_captured)

 @torch.no_grad()
 def evaluate(pool,capture=False):
  model.eval();decoder.eval();memory_decoder.eval();joined=defaultdict(list)
  for start in range(0,len(pool),16):
   _,r=unroll(pool[start:start+16],False,capture)
   for k,v in r.items():joined[k].extend(v)
  joined={k:torch.cat(v) for k,v in joined.items() if v}
  p=joined['probabilities'];y=joined['actions'];ids=joined['ids'];target=joined['outcomes'];logits=joined['logits']
  chosen=joined['risk_probabilities'].argmax(-1);moving=chosen>0;at=chosen[moving]-1
  collision=target[moving,at,0];no_movement=target[moving,at,1]
  recalls=[];memory_recalls=[]
  for c in range(2):
   for val in [0,1]:
    m=target[:,:,c]==val
    if m.any():
     recalls.append(float(((logits[:,:,c]>0)==bool(val))[m].float().mean()))
     memory_recalls.append(float(((joined['memory_logits'][:,:,c]>0)==bool(val))[m].float().mean()))
  groups=defaultdict(list)
  for task,code,profile in zip(joined['tasks'].cpu().tolist(),ids.cpu().tolist(),target.cpu().numpy()):groups[(task,*code)].append(profile)
  alias_errors=sum(np.minimum(np.asarray(v).sum(0),len(v)-np.asarray(v).sum(0)).sum() for v in groups.values())
  metrics=dict(frames=len(y),imitation_accuracy=float((p.argmax(-1)==y).float().mean()),
    imitation_nll=float(-p.gather(1,y[:,None]).clamp_min(1e-20).log().mean()),
    premature_stop_on_movement_examples=float((chosen==0).float().mean()),
    measured_frames=len(target),selected_movement_collision=float(collision.mean()),
    selected_no_movement=float(no_movement.mean()),outcome_balanced_accuracy=float(np.mean(recalls)),
    memory_outcome_balanced_accuracy=float(np.mean(memory_recalls)),
    memory_children_used=[len(set(joined['memory_ids'][:,f].cpu().tolist())) for f in range(model.factors)],
    memory_conclusion_child_difference=float((model.transition[...,::2]-model.transition[...,1::2]).abs().max()),
    action_first_child_difference=float((model.conclusions[:,:,::2]-model.conclusions[:,:,1::2]).abs().max()),
    joint_observed_combinations=len(groups),within_combination_outcome_error=float(alias_errors/target.numel()),
    children_used=[len(set(ids[:,f].cpu().tolist())) for f in range(model.factors)],
    qualification='Teacher-forced component audit on original-training layouts, not closed-loop navigation success.')
  return metrics,joined

 progress(state='caching_parent_states',training_episodes=len(train),probe_episodes=len(probe))
 _,cached=evaluate(train_risk,True)
 head_optimizer=torch.optim.AdamW(memory_decoder.parameters(),lr=1e-3)
 for _ in range(400):
  ix=torch.randint(len(cached['outcomes']),(128,),device='cuda')
  loss=balanced_consequence_loss(memory_decoder(cached['memory_captured'][ix],cached['tasks'][ix]),cached['outcomes'][ix])
  head_optimizer.zero_grad();loss.backward();head_optimizer.step()
 del cached
 assert all(torch.equal(v,source_model[k]) for k,v in model.state_dict().items()),'Auxiliary head warmup altered actor'
 del source_model
 before,_=evaluate(probe);atomic(OUT/'baseline.json',before)
 optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)
 optimizer.load_state_dict(source_optimizer);del source_optimizer
 decoder_optimizer=torch.optim.AdamW(list(decoder.parameters())+list(memory_decoder.parameters()),lr=1e-4)
 started=time.time();history=[];best_score=float('inf')
 for update in range(1,401):
  model.train();decoder.train();memory_decoder.train();batch=rng.sample(train_risk,8)+rng.sample(train,8)
  loss,_=unroll(batch,True);assert torch.isfinite(loss)
  optimizer.zero_grad();decoder_optimizer.zero_grad();loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);torch.nn.utils.clip_grad_norm_(list(decoder.parameters())+list(memory_decoder.parameters()),1.)
  optimizer.step();decoder_optimizer.step()
  if update%10==0:progress(state='training',update=update,loss=float(loss.detach()),seconds=time.time()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
  if update%100==0:
   metrics,_=evaluate(probe);metrics.update(update=update,seconds=time.time()-started);history.append(metrics)
   atomic(OUT/f'probe_{update:04d}.json',metrics)
   payload=dict(model=model.state_dict(),decoder=decoder.state_dict(),memory_decoder=memory_decoder.state_dict(),optimizer=optimizer.state_dict(),decoder_optimizer=decoder_optimizer.state_dict(),manifest=manifest,update=update,metrics=metrics)
   temp=OUT/'checkpoint.partial.pt';torch.save(payload,temp);os.replace(temp,OUT/f'checkpoint_{update:04d}.pt')
   score=metrics['imitation_nll']+.25*metrics['selected_movement_collision']+metrics['premature_stop_on_movement_examples']
   if score<best_score:best_score=score;atomic(OUT/'best_component.json',dict(update=update,score=score,metrics=metrics))
   progress(state='training',update=update,last_component=metrics)
 result=dict(source=warm_commit,baseline=before,history=history,completed_updates=400,time=time.time(),
  initialization=initialization,original_validation_used=False,main_ppo_modified=False,
  qualification='Representation preflight only; do not claim improved navigation without one subsequent heldout rollout evaluation.')
 atomic(OUT/'COMPLETE',result);progress(state='complete',update=400)
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
