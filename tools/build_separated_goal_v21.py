"""Create isolated gradient routing; keep every measured source unchanged."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def change(text,a,b):
    if text.count(a)!=1:raise ValueError((a,text.count(a)))
    return text.replace(a,b)
text=(root/'predictive_nesy/raw_visual_objectives.py').read_text(encoding='utf-8')
text=change(text,'levels=proof_log_support(p,goal,h);planning_losses=[];correct=[]',
    "proof_program=dict(p,transitions=p['transitions'].detach())\n    levels=proof_log_support(proof_program,goal,h);planning_losses=[];correct=[]")
text=change(text,"premise=predicted_premises[t] if proof_from_predicted_state else current[:,t]",
    "premise=(predicted_premises[t] if proof_from_predicted_state else current[:,t]).detach()")
text=change(text,"weights=dict(one_step_dynamics=2.,dynamics=.5,posterior_alignment=.1,reconstruction=.5,subgoal_prediction=1.,\n        teacher_path_goal=.5,proof_action_imitation=proof_weight,rule_inverse=.3,balance=.2,posterior_entropy=.01)",
    "weights=dict(one_step_dynamics=2.,subgoal_prediction=1.,proof_action_imitation=proof_weight)")
text=change(text,'loss=sum(weights[k]*v for k,v in losses.items());metrics={k:float(v.detach()) for k,v in losses.items()}',
    'loss=sum(weight*losses[key] for key,weight in weights.items());metrics={k:float(v.detach()) for k,v in losses.items()}')
text=text[:text.index('\n\ndef transition_objective(model,batch):')]
(root/'predictive_nesy/separated_goal_objectives.py').write_text(text,encoding='utf-8')
text=(root/'predictive_nesy/raw_transition_objectives.py').read_text(encoding='utf-8')
text=change(text,'from .raw_visual_objectives import objective','from .separated_goal_objectives import objective')
text=change(text,'loss=dynamics+.5*open_loop+.1*alignment+.5*reconstruction',
    'loss=dynamics  # Actual observed premise/successor pairs; rollout errors are monitored only.')
(root/'predictive_nesy/separated_transition_objectives.py').write_text(text,encoding='utf-8')
text=(root/'train_replayed_boundary_v20.py').read_text(encoding='utf-8')
text=change(text,'from predictive_nesy.raw_boundary_world import RawBoundaryWorld as SubgoalRuleModel,Config',
    'from predictive_nesy.separated_goal_world import SeparatedGoalWorld as SubgoalRuleModel,Config')
text=change(text,'from predictive_nesy.raw_visual_objectives import objective','from predictive_nesy.separated_goal_objectives import objective')
text=change(text,'from predictive_nesy.raw_transition_objectives import transition_objective','from predictive_nesy.separated_transition_objectives import transition_objective')
text=change(text,'model.load_state_dict(current)',
    "for target_name in list(current):\n        if target_name.startswith('goal_'):\n            source_name='raw_'+target_name[len('goal_'):]\n            if source_name in original['model']:current[target_name]=original['model'][source_name].clone()\n    model.load_state_dict(current)")
text=change(text,"frozen_prefixes=('definition_rules.','subgoal.','raw_')","frozen_prefixes=('definition_rules.','subgoal.','raw_','goal_')")
text=text.replace("'raw_boundary_generated_world_v19'","'separated_rule_goal_world_v21'")
text=change(text,"initialization_scope='Complete v12step1600 model; zero-initialized raw-context branch preserves the initial generated program exactly'",
    "initialization_scope='Complete v20step600; duplicate its raw branch into an independent goal branch with an initially identical program'")
text=change(text,"world_model_intervention='Canonical raw visual histories recovered for each training transition bank; complete rules/subgoal and symbolic runtime otherwise unchanged'",
    "world_model_intervention='Observed one-step transitions train complete rules; independent goal branch receives endpoint and proof-action supervision; proof gradients cannot modify rules'")
text=change(text,"trainable_modules=['definition_rules','subgoal','raw_frame','raw_history','raw_delta'],state_representation_frozen=True,",
    "trainable_modules=['definition_rules','subgoal','raw_frame','raw_history','raw_delta','goal_frame','goal_history','goal_delta'],state_representation_frozen=True,\n        gradient_contract='World: observed one-step transitions only; goal: discrete endpoint CE plus proof action CE through detached rules and premises',\n        monitor_only_losses=['dynamics','teacher_path_goal','rule_inverse','reconstruction','posterior_alignment'],")
text=change(text,"norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)\n        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')",
    "world_norm=torch.nn.utils.clip_grad_norm_([p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('definition_rules.','raw_'))],1.)\n        goal_norm=torch.nn.utils.clip_grad_norm_([p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('subgoal.','goal_'))],1.)\n        norm=torch.maximum(world_norm,goal_norm)\n        metrics['world_gradient_norm']=float(world_norm);metrics['goal_gradient_norm']=float(goal_norm)\n        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')")
text=change(text,"'evaluate_raw_boundary_v19.py'","'evaluate_separated_goal_v21.py'")
(root/'train_separated_goal_v21.py').write_text(text,encoding='utf-8')
text=(root/'evaluate_raw_boundary_v19.py').read_text(encoding='utf-8')
text=change(text,'from predictive_nesy.raw_boundary_world import RawBoundaryWorld as SubgoalRuleModel,Config,export_program',
    'from predictive_nesy.separated_goal_world import SeparatedGoalWorld as SubgoalRuleModel,Config,export_program')
text=text.replace("'raw_boundary_generated_world_v19'","'separated_rule_goal_world_v21'")
(root/'evaluate_separated_goal_v21.py').write_text(text,encoding='utf-8')
print('Created isolated v21 model, losses, trainer and evaluator')
