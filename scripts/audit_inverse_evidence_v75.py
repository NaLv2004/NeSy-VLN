"""Intervene on action and observation inputs to the trained inverse objective.

Uses only saved original-training integer tokens, no rendering or validation.
The source is a terminal full-sequence FSM, not either live v74 trainer.
"""
import json,os,random,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
B=Path('/home/user/xjy/nesy_vla_20261010')
sys.path.insert(0,str(B/'probabilistic_automaton_v56/src'));sys.path.insert(1,str(B/'symbolic_paired_v12'))
from nesy_vla.runtime import read,atomic,setup,DataBank
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton

def main():
 out=B/'inverse_evidence_v75';out.mkdir(exist_ok=True)
 assert not (out/'result.json').exists(),'Completed diagnostic must not be repeated'
 torch.set_num_threads(2);torch.manual_seed(75)
 cfg=read(B/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
 from stripsnav.io import load_committed
 source=B/'probabilistic_automaton_v56/soft_recurrent/step_000040'
 payload=load_committed(source);model=ProbabilisticTokenAutomaton(**payload['manifest']['architecture']).cuda().eval()
 model.load_state_dict(payload['model'],strict=True);del payload
 old=B/'spatial_automaton_v43';extra=B/'data/onpolicy_recovery_v48'
 rows=read(old/'training_rows.json')+read(extra/'training_rows.json')
 codes=np.concatenate([np.load(old/'visual_tokens.npy'),np.load(extra/'visual_tokens.npy')])
 assert len(rows)==len(codes)
 groups={}
 for i,row in enumerate(rows):
  assert bank.by_id[row.get('source_episode_id',row['episode_id'])]['scene'] in bank.provenance['train_layouts']
  groups.setdefault(row['episode_id'],[]).append(i)
 selected=list(groups.items());random.Random(75).shuffle(selected);selected=selected[:128]
 names=['actual','previous_observation','constant_observation','counterfactual_action']
 totals={k:{'correct_true':0,'correct_counterfactual':0,'changed_prediction':0,'tv_sum':0.} for k in names}
 counts=0;examples=[];proof_error=0.;started=time.time()
 with torch.no_grad():
  for offset in range(0,len(selected),8):
   batch=selected[offset:offset+8];n=len(batch);length=max(len(i) for _,i in batch)
   idx=torch.zeros(n,length,dtype=torch.long,device='cuda');valid=torch.zeros_like(idx,dtype=torch.bool)
   for j,(_,indices) in enumerate(batch):idx[j,:len(indices)]=torch.tensor(indices,device='cuda');valid[j,:len(indices)]=True
   allcodes=torch.tensor(codes.astype(np.int64),device='cuda');x=allcodes[idx]
   y=torch.tensor([r['label'] for r in rows],device='cuda')[idx]
   task=torch.tensor([['chair','table'].index(rows[i[0]]['category']) for _,i in batch],device='cuda')
   _,prior=model.initial_state(n,'cuda');previous_action=torch.full((n,),17,dtype=torch.long,device='cuda')
   for step in range(length):
    if step%10==0:program=model.generate(task,x[:,max(0,step-10):step+1],step)
    r=model.step(program,x[:,step],prior,previous_action,step)
    proof=model.proof(program,r['observed_ids'],prior,previous_action,r['state_values'],step)
    proof_error=max(proof_error,float((proof['probabilities']-r['probabilities']).abs().max()));assert proof_error==0
    if step:
     active=valid[:,step];count=int(active.sum());counts+=count
     fake=previous_action.remainder(16)+1
     previous_observed=model.step(program,x[:,step-1],prior,previous_action,step)['state_values']
     constant=F.one_hot(torch.zeros_like(r['observed_ids']),model.states).float()
     changed_obs=model.transition_belief(program,constant,prior,previous_action,step)
     changed_action=model.transition_belief(program,r['observed_values'],prior,fake,step)
     state_variants={'actual':r['state_values'],'previous_observation':previous_observed,'constant_observation':changed_obs,'counterfactual_action':changed_action}
     probabilities={k:model.inverse_action(prior,v).softmax(-1) for k,v in state_variants.items()}
     actual=probabilities['actual']
     for name,prob in probabilities.items():
      totals[name]['correct_true']+=int(((prob.argmax(-1)==previous_action)&active).sum())
      totals[name]['correct_counterfactual']+=int(((prob.argmax(-1)==fake)&active).sum())
      totals[name]['changed_prediction']+=int(((prob.argmax(-1)!=actual.argmax(-1))&active).sum())
      totals[name]['tv_sum']+=float(((prob-actual).abs().sum(-1)*.5*active).sum())
     for j in range(n):
      if active[j] and len(examples)<24 and probabilities['counterfactual_action'][j].argmax()!=actual[j].argmax():
       examples.append({'episode':batch[j][0],'step':step,'true_action':int(previous_action[j]),'injected_action':int(fake[j]),
          'predicted_actual':int(actual[j].argmax()),'predicted_when_observation_unchanged':int(probabilities['previous_observation'][j].argmax()),
          'predicted_when_action_changed':int(probabilities['counterfactual_action'][j].argmax())})
    prior=r['state_values'];previous_action=y[:,step]
 result={'time':time.time(),'source_checkpoint':str(source),'source_commit':read(source/'COMMITTED'),
  'original_training_only':True,'development_or_fixed80_used':False,'fresh_simulator_rollouts':0,
  'source_episode_ids':[e for e,_ in selected],'transition_count':counts,'seconds':time.time()-started,
  'action_rule_proof_max_error':proof_error,'interventions':{k:dict(v,accuracy_true=v['correct_true']/counts,
      accuracy_injected=v['correct_counterfactual']/counts,changed_prediction_fraction=v['changed_prediction']/counts,
      mean_probability_tv=v['tv_sum']/counts) for k,v in totals.items()},'examples':examples,
  'qualification':'Frozen trained inverse head reads recurrent beliefs whose update explicitly includes the action being predicted. Interventions measure shortcut sensitivity, not learned-navigation success. Correcting inverse supervision still requires matched training.'}
 atomic(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_episode_ids','examples']},indent=2))
if __name__=='__main__':main()
