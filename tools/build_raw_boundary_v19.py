"""Create explicit source copies; retain all earlier experiment implementations."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'

def change(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old));return text.replace(old,new)

text=(root/'predictive_nesy/visual_graph_subgoal_training.py').read_text(encoding='utf-8')
text=change(text,'proof_only_if_goal_correct=False):','proof_only_if_goal_correct=False,program=None):')
text=change(text,"    p=model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)",
"    p=program if program is not None else model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)")
(root/'predictive_nesy/raw_visual_objectives.py').write_text(text,encoding='utf-8')
text=(root/'predictive_nesy/visual_memory_subgoal_training.py').read_text(encoding='utf-8')
text=change(text,'from .visual_graph_subgoal_training import objective','from .raw_visual_objectives import objective')
text=change(text,'state_separation_weight=0.):','state_separation_weight=0.,program=None):')
text=change(text,"    p=model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=batch['start'])",
"    p=program if program is not None else model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=batch['start'])")
(root/'predictive_nesy/raw_transition_objectives.py').write_text(text,encoding='utf-8')

text=(root/'train_definition_world_v12.py').read_text(encoding='utf-8')
text=change(text,'from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld as SubgoalRuleModel,Config',
'''from predictive_nesy.raw_boundary_world import RawBoundaryWorld as SubgoalRuleModel,Config
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures''')
text=change(text,'from predictive_nesy.rollout_proof_training import objective,transition_objective',
'''from predictive_nesy.raw_visual_objectives import objective
from predictive_nesy.raw_transition_objectives import transition_objective''')
text=change(text,"    def forward(self,data,kind):return objective(self.model,data) if kind=='demonstrations' else transition_objective(self.model,data)",
'''    def forward(self,data,kind):
        features=data.pop('boundary_features');available=data.pop('boundary_available')
        if kind=='demonstrations':
            start=int(data['start'][0]);width=min(start+1,11);lengths=data['history_length']
        else:
            start=data['start'];width=data['history'].shape[1]
            lengths=torch.full((len(data['task']),),width,device=data['task'].device,dtype=torch.long)
        generated=self.model.generate(data['task'],data['history'][:,:width],data['previous_actions'][:,:width],lengths,
            start=start,boundary_features=features,boundary_available=available)
        if kind=='demonstrations':return objective(self.model,data,proof_from_predicted_state=True,program=generated)
        loss,metrics=transition_objective(self.model,data,program=generated)
        metrics['counterfactual_raw_boundary_available']=float(available.float().mean())
        return loss,metrics''')
text=change(text,"    frozen_prefixes=('definition_rules.','subgoal.')","    frozen_prefixes=('definition_rules.','subgoal.','raw_')")
text=change(text,"    sources=[TransitionBank(path,bank.provenance['train_layouts']) for path in a.transitions]",
"    features=RawBoundaryFeatures(BASE/'predictive_rules_20261015/prequant_visual_features_v18',device)\n    sources=[TransitionBank(path,bank.provenance['train_layouts']) for path in a.transitions]")
text=change(text,"architecture='anonymous_definition_conditioned_world_v12'","architecture='raw_boundary_generated_world_v19'")
text=change(text,"initialization_scope='complete v11 representation and goal head; new definition-conditioned rule decoder initialized by training-only distillation'",
"initialization_scope='Complete v12step1600 model; zero-initialized raw-context branch preserves the initial generated program exactly'")
text=change(text,"world_model_intervention='complete transition rules conditioned on learned source and destination definition vectors; no manually assigned semantics'",
"world_model_intervention='Raw CNN/PCA boundary evidence before patch quantization conditions complete generated rules and a discrete subgoal; runtime remains symbolic',raw_boundary_input=features.manifest")
text=change(text,"trainable_modules=['definition_rules','subgoal']","trainable_modules=['definition_rules','subgoal','raw_frame','raw_history','raw_delta']")
text=change(text,"        data=batch(rng.choices(by_start[start],k=a.batch_size),device)",
"        selected=rng.choices(by_start[start],k=a.batch_size);data=batch(selected,device)\n        data['boundary_features'],data['boundary_available']=features.demonstrations(selected)")
text=change(text,"            data=source.sample(rng,a.transition_batch_size,device)",
"            data=source.sample(rng,a.transition_batch_size,device)\n            data['boundary_features'],data['boundary_available']=features.transitions(data)")
text=change(text,"'evaluate_definition_world_v12.py'","'evaluate_raw_boundary_v19.py'")
(root/'train_raw_boundary_v19.py').write_text(text,encoding='utf-8')

text=(root/'evaluate_definition_world_v12.py').read_text(encoding='utf-8')
text=change(text,'from predictive_nesy.fast_vision import BoundaryGraphEncoder','from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder as BoundaryGraphEncoder')
text=change(text,'from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld as SubgoalRuleModel,Config,export_program',
    'from predictive_nesy.raw_boundary_world import RawBoundaryWorld as SubgoalRuleModel,Config,export_program')
text=change(text,"                            history=torch.stack([encoder.encode(images[i][-width:]) for i in active])",
"                            encoded=[encoder.encode_joint(images[i][-width:]) for i in active]\n                            history=torch.stack([item[0] for item in encoded]);raw=torch.stack([item[1] for item in encoded])")
text=change(text,"generated=model.generate(tasks[active],history,previous,lengths,start=step)",
"generated=model.generate(tasks[active],history,previous,lengths,start=step,boundary_features=raw)")
text=change(text,"'anonymous_definition_conditioned_world_v12'","'raw_boundary_generated_world_v19'")
(root/'evaluate_raw_boundary_v19.py').write_text(text,encoding='utf-8')
print('Created raw-boundary objective, training and evaluation source copies.')
