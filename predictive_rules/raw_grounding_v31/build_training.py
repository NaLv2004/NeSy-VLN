"""Build the new isolated trainer from the preserved hard-planning trainer."""
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root.parent/'stable_rules_v30/train.py').read_text(encoding='utf-8')
def replace(old,new):
    global text
    if text.count(old)!=1:raise ValueError((old,text.count(old)))
    text=text.replace(old,new)
replace('from planning_model import PlanningObservedWorld as AdaptiveObservedWorld, Config',
        'from raw_model import RawObservedWorld as AdaptiveObservedWorld, Config')
replace('from stable_objectives import observed_objective, branch_representation_loss',
        'from raw_objectives import demonstration_objective, branch_objective\nfrom raw_data import RawFeatures, RawBranchBank\nfrom predictive_nesy.environment import make_bank')
replace("    return model.prepare(data['task'],data['history'][:,:width],data['previous_actions'][:,:width],\n                         data['history_length'],start=start,boundary_features=raw,boundary_available=available)",
        "    p=model.prepare(data['task'],data['history'][:,:width],data['previous_actions'][:,:width],\n                         data['history_length'],start=start,boundary_features=raw,boundary_available=available)\n    p['_targets_visual']=features.targets(rows)\n    return p")
replace("assert not unexpected and all(k.startswith(('revision.','correction.','inverse_discrete.')) for k in missing)",
        "assert not unexpected and all(k.startswith(('raw_observation.','raw_definition_context.')) for k in missing)")
replace("                             'context_read.','history.','context_norm.','task.','action_embed.')",
        "                             'context_read.','history.','context_norm.','task.','action_embed.','raw_observation.','raw_definition_context.')")
replace("    parameters=('definition_rules.','subgoal.','raw_','goal_','rule_evidence.','revision.','correction.')\n    for name,p in model.named_parameters():p.requires_grad_(name.startswith(parameters))\n    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}",
        "    for parameter in model.parameters():parameter.requires_grad_(True)")
replace("    features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cuda')\n    banks=[TransitionBank(x['path'],old['manifest']['data']['train_layouts']) for x in old['manifest']['transition_banks']]",
        "    features=RawFeatures(ROOT/'prequant_visual_features_v18','cuda')\n    branches_bank=RawBranchBank(ROOT/'raw_successor_features_v31',old['manifest']['data']['train_layouts'])\n    episode_lookup=make_bank().by_id\n    if len(branches_bank.keys)<512:raise RuntimeError('Training launch requires at least512 aligned branch positions')")
replace("architecture='stable_observed_rules_v30'","architecture='raw_observed_joint_v31'")
replace("learned_semantic_state_labels=False,initial_perception_frozen=True,frontend_CNN_frozen=True,",
        "learned_semantic_state_labels=False,initial_perception_frozen=False,frontend_CNN_frozen=True,")
replace("representation_training='Stable learned v29 definitions, visual encoder and history; complete rule/goal generation and adaptive renewal are trained on all existing training-only data',\n        definitions_frozen=True,causal_state_context_frozen=True,stationary_anonymous_labels=True,\n        perception_proof_weight=0.,demonstration_inverse_weight=0.,branch_batch_size=128,",
        "representation_training='Unquantized 49x128 visual evidence jointly trains the categorical encoder, definitions, world model and goal generator; only512-way argmax IDs reach the planner',\n        definitions_frozen=False,causal_state_context_frozen=False,stationary_anonymous_labels=False,\n        perception_proof_weight=.25,demonstration_inverse_weight=.1,branch_batch_size=64,")
replace("inverse_used_at_runtime=False,ema_decay=.99,training_only_inverse_head_frozen=True,",
        "inverse_used_at_runtime=False,ema_decay=.995,training_only_inverse_head_frozen=False,")
replace("transition_banks=[dict(path=str(x.directory),**x.refresh()) for x in banks])",
        "transition_banks=old['manifest']['transition_banks'],raw_successor_cache=str(ROOT/'raw_successor_features_v31'),\n        raw_successor_available=branches_bank.refresh())")
replace("        with torch.no_grad():\n            teacher_program=prepare(teacher,data,features,rows)\n            targets=teacher.training_targets(teacher_program,data['targets'],data['actions']).argmax(-1).squeeze(-1)\n        with torch.autocast('cuda',dtype=torch.bfloat16):\n            loss,metrics,labels=observed_objective(model,data,p,target_labels=targets)",
        "        with torch.no_grad():teacher_program=prepare(teacher,data,features,rows)\n        with torch.autocast('cuda',dtype=torch.bfloat16):\n            loss,metrics,labels=demonstration_objective(model,teacher,data,p,teacher_program)")
replace("                symbols=model.encode_sequence(old_p,prefix)[0].argmax(-1)",
        "                prefix_ids=torch.tensor([features.groups[name][old_start:start+1] for name,_ in keys],device='cuda')\n                prefix_visual=features.features[prefix_ids]\n                symbols=model.encode_sequence(old_p,prefix,prefix_visual)[0].argmax(-1)")
replace("                old_next=model.encode_sequence(old_p,torch.cat([prefix,next_codes[:,None]],1))[0][:,-1].argmax(-1)",
        "                next_ids=torch.tensor([features.groups[name][min(start+1,len(features.groups[name])-1)] for name,_ in keys],device='cuda')\n                old_next=model.encode_sequence(old_p,torch.cat([prefix,next_codes[:,None]],1),\n                    torch.cat([prefix_visual,features.features[next_ids,None]],1))[0][:,-1].argmax(-1)")
start=text.index('        selected=rng.choices(banks,')
end=text.index('        if not torch.isfinite(branch_loss)',start)
text=text[:start]+'''        if update%100==1:branches_bank.refresh()
        branches=branches_bank.sample(rng,64,'cuda',episode_lookup)
        raw=branches['boundary_visual'];width=branches['history'].shape[1]
        lengths=torch.full((len(branches['task']),),width,device='cuda',dtype=torch.long)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            branch_program=model.prepare(branches['task'],branches['history'],branches['previous_actions'],lengths,
                start=branches['start'],boundary_features=raw)
            with torch.no_grad():
                teacher_branch=teacher.prepare(branches['task'],branches['history'],branches['previous_actions'],lengths,
                    start=branches['start'],boundary_features=raw)
            branch_loss,branch_metrics=branch_objective(model,teacher,branch_program,teacher_branch,branches)
''' + text[end:]
replace("groups=dict(world=('definition_rules.','raw_','rule_evidence.'),goal=('subgoal.','goal_'),",
        "groups=dict(world=('definition_rules.','raw_frame.','raw_history.','raw_delta.','rule_evidence.'),goal=('subgoal.','goal_'),")
replace("perception=('tokens.','observation.','visual_memory.','memory_projection.'),",
        "perception=('tokens.','observation.','visual_memory.','memory_projection.','raw_observation.'),")
replace("context=('context_read.','history.','context_norm.','task.','action_embed.'),inverse=('inverse_discrete.',))",
        "context=('context_read.','history.','context_norm.','task.','action_embed.','raw_definition_context.'),inverse=('inverse_discrete.',))")
replace("value.lerp_(online[name],.01)","value.lerp_(online[name],.005)")
replace("memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,representation_frozen=True,branch_batch_size=128)",
        "memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,representation_frozen=False,branch_batch_size=64,\n                    raw_branch_positions_available=len(branches_bank.keys))")
replace("    assert all(torch.equal(value,dict(model.named_parameters())[name].detach().cpu()) for name,value in frozen.items())\n",'')
(root/'train.py').write_text(text,encoding='utf-8')

text=(root.parent/'stable_rules_v30/evaluate.py').read_text(encoding='utf-8')
replace('from planning_model import PlanningObservedWorld as AdaptiveObservedWorld, Config',
        'from raw_model import RawObservedWorld as AdaptiveObservedWorld, Config')
replace("=='stable_observed_rules_v30'","=='raw_observed_joint_v31'")
replace("                                feature=model.observation(model.embed(code))\n                                memory,last,old_codes=model.advance(perception['ground_memory'],perception['ground_last'],perception['ground_codes'],feature,code)\n                                perception.update(ground_memory=memory,ground_last=last,ground_codes=old_codes)\n                                observed=int(model.ground(perception,code).argmax(-1))",
        "                                observed=int(model.observe(perception,code,raw).argmax(-1))")
replace("['definitions','ground_memory','ground_last','ground_codes']",
        "['definitions','ground_memory','ground_last','ground_codes','ground_visual']")
(root/'evaluate.py').write_text(text,encoding='utf-8')
print('Generated isolated raw-grounding trainer and matched runtime.')
