"""Build the single follow-up that teaches the duplicated memory conclusions."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'scripts/train_consequence_refinement_v93.py').read_text(encoding='utf-8')
def replace(old,new,count=1):
 global s
 assert s.count(old)==count,(old,s.count(old))
 s=s.replace(old,new)
replace("OUT=B/'consequence_refinement_v93'","OUT=B/'memory_consequence_v97'")
replace('balanced_consequence_loss,merge_child_belief','balanced_consequence_loss,merge_child_belief,discrete_memory_equalities')
start=s.index(" warm=B/'parallel_ppo_v85/best_development'")
end=s.index(' manifest=dict(',start)
s=s[:start]+''' warm=B/'consequence_refinement_v93';checkpoint_file=warm/'checkpoint_0400.pt'
 assert sha(checkpoint_file)=='36e80f961febe8d8c9f4d5f7a4b94f4f2a6199ffd8360309bc6a25a35b77e4e4'
 assert read(warm/'COMPLETE')['completed_updates']==400
 saved=torch.load(checkpoint_file,map_location='cpu',weights_only=False)
 warm_commit=dict(checkpoint_file=checkpoint_file.name,checkpoint_sha256=sha(checkpoint_file),epoch=400)
 architecture=saved['manifest']['architecture'];assert architecture['states']==32
 model=ProbabilisticTokenAutomaton(**architecture).cuda();model.load_state_dict(saved['model'])
 decoder=ConsequenceDecoder(model.factors,model.states).cuda();decoder.load_state_dict(saved['decoder'])
 memory_decoder=ConsequenceDecoder(model.factors,model.states).cuda()
 source_optimizer=saved['optimizer'];source_model={k:v.detach().clone() for k,v in model.state_dict().items()}
'''+s[end:]
replace('consequence_weight=.5,rule_execution_cost_weight=.2,head_warmup_batches=400,','consequence_weight=.5,memory_consequence_weight=.5,rule_execution_cost_weight=.2,head_warmup_batches=400,')
replace("initialization='Each of16 anonymous states becomes2 child states with mathematically identical merged transitions and action rules. No state meaning supplied.'","initialization='Resume exact v93 actor, observation decoder and actor optimizer. New training-only decoder receives hard anonymous memory IDs; its gradients break identical sibling transition conclusions.'")
replace("teacher_signals='Existing counterfactual executed collision/no-movement outcomes only train anonymous equality grounding. Outcome decoder discarded at deployment; no runtime action masks.'","teacher_signals='Existing measured outcomes supervise both hard observed equalities and hard memory equalities. Both auxiliary decoders discarded at deployment. No manually assigned state meanings or runtime action masks.'")
start=s.index(' @torch.no_grad()\n def check_initialization(')
end=s.index(' def unroll(',start)
s=s[:start]+''' initialization=dict(source_sha256=sha(checkpoint_file),source_update=400,state_count=32,
  verification='Strict source actor and decoder loading; memory head does not participate in action selection.')
 atomic(OUT/'initialization_verification.json',initialization)

'''+s[end:]
replace('total=x.new_zeros((),dtype=torch.float);captured=[];logits_all=[];outcomes=[];ids=[];tasks_all=[];probs=[];actions=[];expected=[];risk_probs=[];risk_actions=[]',
 'total=x.new_zeros((),dtype=torch.float);captured=[];logits_all=[];outcomes=[];ids=[];tasks_all=[];probs=[];actions=[];expected=[];risk_probs=[];risk_actions=[];memory_logits=[];memory_ids=[];memory_captured=[]')
replace("hard=r['observed_values'][m];prediction=decoder(hard,task[m]);target=targets[ix[:,step][m]]", "hard=r['observed_values'][m];prediction=decoder(hard,task[m]);target=targets[ix[:,step][m]]\n    hard_memory=discrete_memory_equalities(r['state_values'][m]);memory_prediction=memory_decoder(hard_memory,task[m])\n    memory_logits.append(memory_prediction.detach());memory_ids.append(r['state_ids'][m].detach())")
replace('total=total+.5*balanced_consequence_loss(prediction,target)*m.sum()/risk_count','total=total+.5*balanced_consequence_loss(prediction,target)*m.sum()/risk_count\n     total=total+.5*balanced_consequence_loss(memory_prediction,target)*m.sum()/risk_count')
replace('if capture:captured.append(hard.detach())','if capture:captured.append(hard.detach());memory_captured.append(hard_memory.detach())')
replace('return total,dict(logits=logits_all,outcomes=outcomes,ids=ids,captured=captured,tasks=tasks_all,probabilities=probs,actions=actions,risk_probabilities=risk_probs)',
 'return total,dict(logits=logits_all,outcomes=outcomes,ids=ids,captured=captured,tasks=tasks_all,probabilities=probs,actions=actions,risk_probabilities=risk_probs,memory_logits=memory_logits,memory_ids=memory_ids,memory_captured=memory_captured)')
replace('model.eval();decoder.eval();joined=defaultdict(list)','model.eval();decoder.eval();memory_decoder.eval();joined=defaultdict(list)')
replace('recalls=[]','recalls=[];memory_recalls=[]')
replace("if m.any():recalls.append(float(((logits[:,:,c]>0)==bool(val))[m].float().mean()))", "if m.any():\n     recalls.append(float(((logits[:,:,c]>0)==bool(val))[m].float().mean()))\n     memory_recalls.append(float(((joined['memory_logits'][:,:,c]>0)==bool(val))[m].float().mean()))")
replace("selected_no_movement=float(no_movement.mean()),outcome_balanced_accuracy=float(np.mean(recalls)),", "selected_no_movement=float(no_movement.mean()),outcome_balanced_accuracy=float(np.mean(recalls)),\n    memory_outcome_balanced_accuracy=float(np.mean(memory_recalls)),\n    memory_children_used=[len(set(joined['memory_ids'][:,f].cpu().tolist())) for f in range(model.factors)],\n    memory_conclusion_child_difference=float((model.transition[...,::2]-model.transition[...,1::2]).abs().max()),\n    action_first_child_difference=float((model.conclusions[:,:,::2]-model.conclusions[:,:,1::2]).abs().max()),")
replace("head_optimizer=torch.optim.AdamW(decoder.parameters(),lr=1e-3)","head_optimizer=torch.optim.AdamW(memory_decoder.parameters(),lr=1e-3)")
replace("loss=balanced_consequence_loss(decoder(cached['captured'][ix],cached['tasks'][ix]),cached['outcomes'][ix])", "loss=balanced_consequence_loss(memory_decoder(cached['memory_captured'][ix],cached['tasks'][ix]),cached['outcomes'][ix])")
replace('del cached\n before,_=evaluate(probe);', "del cached\n assert all(torch.equal(v,source_model[k]) for k,v in model.state_dict().items()),'Auxiliary head warmup altered actor'\n del source_model\n before,_=evaluate(probe);")
replace("optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)","optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)\n optimizer.load_state_dict(source_optimizer);del source_optimizer")
replace('decoder_optimizer=torch.optim.AdamW(decoder.parameters(),lr=1e-4)', 'decoder_optimizer=torch.optim.AdamW(list(decoder.parameters())+list(memory_decoder.parameters()),lr=1e-4)')
replace('model.train();decoder.train();batch=', 'model.train();decoder.train();memory_decoder.train();batch=')
replace('torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.)','torch.nn.utils.clip_grad_norm_(list(decoder.parameters())+list(memory_decoder.parameters()),1.)')
replace('payload=dict(model=model.state_dict(),decoder=decoder.state_dict(),optimizer=', 'payload=dict(model=model.state_dict(),decoder=decoder.state_dict(),memory_decoder=memory_decoder.state_dict(),optimizer=')
compile(s,'train_memory_consequence_v97.py','exec')
(ROOT/'scripts/train_memory_consequence_v97.py').write_text(s,encoding='utf-8')
print('Built a source-resuming memory-consequence trainer. Not launched.')
