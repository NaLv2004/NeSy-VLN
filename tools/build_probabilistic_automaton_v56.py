"""Matched soft-state experiments, preserving integer observation premises."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'scripts/train_goal_conditioned_automaton_v55.py').read_text()
edits={
 "from nesy_vla.factorized_automaton import FactorizedTokenAutomaton":"from nesy_vla.factorized_automaton import FactorizedTokenAutomaton\nfrom nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton",
 "choices=['bridge_only','goal_programs']":"choices=['soft_recurrent','soft_reset']",
 "out=base/'goal_conditioned_automaton_v55'/args.arm":"out=base/'probabilistic_automaton_v56'/args.arm",
 "'reset_memory':False,'context_bridge':True,'task_banks':args.arm=='goal_programs'":"'reset_memory':args.arm=='soft_reset','context_bridge':True,'task_banks':True",
 "model=FactorizedTokenAutomaton(**architecture).cuda()":"model=ProbabilisticTokenAutomaton(**architecture).cuda()",
 " assert torch.equal(before['probabilities'],after['probabilities'])":" # Soft transition marginalization intentionally changes initial actions.\n # The observation predicates and all transferred weights remain identical.",
 "'initial_function_max_error':initial_error":"'initial_hard_to_soft_action_probability_difference':initial_error",
 "proof=model.proof(program,result['observed_ids'],memory_ids,last_action,result['state_ids'],step)":"proof=model.proof(program,result['observed_ids'],memory,last_action,result['state_values'],step)",
 "'previous_state':memory_ids[i].tolist(),'previous_action':int(last_action[i]),'state':result['state_ids'][i].tolist(),":"'previous_state':memory_ids[i].tolist(),'previous_action':int(last_action[i]),'state':result['state_ids'][i].tolist(),\n                     'previous_belief':memory[i].tolist(),'state_belief':result['state_values'][i].tolist(),",
 "seen+=(r['state_values'].detach()*active[:,None,None]).sum(0)":"seen+=(F.one_hot(r['state_ids'],model.states).float()*active[:,None,None]).sum(0)",
 "'all_actions_have_discrete_transition_and_rule_proofs':True":"'all_actions_have_discrete_transition_and_rule_proofs':True,'state_semantics':'Soft beliefs over finite learned state IDs; exact transition marginalization and product-conjunction action rules.'",
 "'goal_adaptation':'Zero-initialized non-saturating goal/history bridge; treatment additionally learns goal-indexed definition/transition/conclusion banks. No predicate meanings or transitions are supplied.'":"'goal_adaptation':'Learned goal-indexed definition/transition/conclusion banks plus goal/history program generator.','state_semantics':'Hard learned observation IDs; soft categorical state beliefs. Transitions marginalize explicit learned tables and action rules use product conjunction. No visual features in the interpreter.','comparison':'Same v51 recurrent20 source and data as v55, fresh optimizers. Soft recurrent versus resetting memory; v55 hard-state arm is the existing control. Initial action distributions are intentionally different.'",
}
for old,new in edits.items():
 assert old in text,old
 text=text.replace(old,new)
end=text.index('"""',3)+3
text='"""Learn generated soft finite-state programs from integer-token demonstrations.\n\nNo manual predicate definitions, no map, no continuous visual action bypass.\n"""'+text[end:]
(root/'scripts/train_probabilistic_automaton_v56.py').write_text(text,encoding='utf-8')
print('Created probability-state trainer; exact categorical marginalization is audited in every rollout.')
