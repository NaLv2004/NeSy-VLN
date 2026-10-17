"""Create matched goal-preservation experiments from the audited v51 trainer."""
from pathlib import Path
root=Path(__file__).resolve().parents[1];text=(root/'scripts/train_factorized_automaton_v51.py').read_text()
edits={
 "choices=['recurrent','reset_memory']":"choices=['bridge_only','goal_programs']",
 "out=base/'factorized_automaton_v51'/args.arm":"out=base/'goal_conditioned_automaton_v55'/args.arm",
 "'rank':4,'reset_memory':args.arm=='reset_memory'":"'rank':4,'reset_memory':False,'context_bridge':True,'task_banks':args.arm=='goal_programs'",
 "model=FactorizedTokenAutomaton(**architecture).cuda()":
 "model=FactorizedTokenAutomaton(**architecture).cuda()\nwarm_source=base/'factorized_automaton_v51/recurrent/step_000020'\nwarm=load_committed(warm_source)\nmodel.initialize_from_shared(warm['model'])\nsource_model=FactorizedTokenAutomaton(**warm['manifest']['architecture']).cuda().eval()\nsource_model.load_state_dict(warm['model'])\nwith torch.no_grad():\n sample=torch.from_numpy(codes[:64].astype(np.int64)).cuda();goals=torch.arange(64,device='cuda')%2\n _,memory=model.initial_state(64,'cuda');prior=torch.full((64,),17,device='cuda',dtype=torch.long)\n before=source_model.step(source_model.generate(goals,sample[:,None],0),sample,memory,prior,0)\n after=model.step(model.generate(goals,sample[:,None],0),sample,memory,prior,0)\n assert torch.equal(before['probabilities'],after['probabilities'])\n assert torch.equal(before['observed_ids'],after['observed_ids'])\n initial_error=float((before['probabilities']-after['probabilities']).abs().max())\ndel source_model,warm;gc.collect();torch.cuda.empty_cache()",
 "manifest={'architecture':architecture,'source_commit':source_commit":
 "manifest={'architecture':architecture,'warm_start_checkpoint':str(warm_source),'warm_start_commit':read(warm_source/'COMMITTED'),'initial_function_max_error':initial_error,'optimizer_resume':False,'source_commit':source_commit",
 "'teacher_forced_training_accuracy_is_not_navigation_success':True}":
 "'teacher_forced_training_accuracy_is_not_navigation_success':True,'goal_adaptation':'Zero-initialized non-saturating goal/history bridge; treatment additionally learns goal-indexed definition/transition/conclusion banks. No predicate meanings or transitions are supplied.'}",
 "save(out/'latest',epoch);append(out/'epochs.jsonl',record);progress(state='training',**record)":
 "record['paired_goal_audit']=paired_goal_audit()\n        save(out/'latest',epoch);append(out/'epochs.jsonl',record);progress(state='training',**record)",
}
for old,new in edits.items():
 assert old in text,old
 text=text.replace(old,new)
needle='vision=None;sims=[];vision_cfg=None;vocabulary=None'
audit='''first_rows={}
for i,row in enumerate(rows):first_rows.setdefault(row['episode_id'],i)
goal_pairs=[]
for group in bank.paired_demo_groups:
 i,j=[first_rows[eid] for eid in group]
 if rows[i]['label']!=rows[j]['label']:goal_pairs.append((i,j))
assert len(goal_pairs)==55
@torch.no_grad()
def paired_goal_audit():
 inputs=[];goals=[];targets=[]
 for i,j in goal_pairs:
  inputs.extend([i,i]);goals.extend([tasks.index(rows[i]['category']),tasks.index(rows[j]['category'])]);targets.extend([rows[i]['label'],rows[j]['label']])
 x=all_codes[torch.tensor(inputs,device='cuda')];goal=torch.tensor(goals,device='cuda')
 _,prior=model.initial_state(len(x),'cuda');program=model.generate(goal,x[:,None],0)
 result=model.step(program,x,prior,torch.full((len(x),),17,device='cuda',dtype=torch.long),0)
 p=result['probabilities'];actions=p.argmax(-1)
 return {'pairs':len(goal_pairs),'changed_actions':int((actions[::2]!=actions[1::2]).sum()),
  'matched_demonstrations':int((actions==torch.tensor(targets,device='cuda')).sum()),'total_actions':len(targets),
  'mean_probability_total_variation':float((p[::2]-p[1::2]).abs().sum(-1).mean()/2),
  'qualification':'Training-pair goal sensitivity, not navigation success.'}

'''
assert needle in text;text=text.replace(needle,audit+needle)
text=text.replace('"""Train learned categorical state machines from immutable integer-token demos.',
                  '"""Repair measured goal erasure in learned categorical state machines.')
(root/'scripts/train_goal_conditioned_automaton_v55.py').write_text(text)
print('Created goal-conditioned trainer with exact function-preserving warm-start check')
