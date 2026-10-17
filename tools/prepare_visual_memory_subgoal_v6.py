"""Build isolated entry points while leaving previous run snapshots untouched."""
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'predictive_rules'
train=(root/'train_visual_graph_subgoal.py').read_text(encoding='utf-8')
train=train.replace('from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel as SubgoalRuleModel,Config',
    'from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel as SubgoalRuleModel,Config')
train=train.replace('from predictive_nesy.visual_graph_subgoal_training import objective,transition_objective',
    'from predictive_nesy.visual_memory_subgoal_training import objective,transition_objective')
old="preserve=name=='offsets' or name.startswith(('tokens.','observation.','context_read.','history.','task.','action_embed.','context_norm.'))"
assert train.count(old)==1
train=train.replace(old,'preserve=True  # Same v5 modules; new visual memory starts with zero output influence')
train=train.replace("anonymous_visual_graph_subgoal_v5","anonymous_visual_memory_subgoal_v6")
train=train.replace("initialization_scope='visual encoder and boundary history; definitions, transitions, subgoal and reconstruction initialized fresh'",
    "initialization_scope='full v5 checkpoint plus zero-output visual-history memory; optimizer initialized fresh'")
train=train.replace("visual_state_labels_ignore_future_actions=True,teacher_forced_dynamics_supervision=True,",
    "visual_state_labels_ignore_future_actions=True,teacher_forced_dynamics_supervision=True,\n        repeated_image_leaves_state_unchanged=True,visual_history_grounding_only_at_rule_boundaries=True,")
train=train.replace('evaluate_visual_graph_subgoal.py','evaluate_visual_memory_subgoal.py')
(root/'train_visual_memory_subgoal.py').write_text(train,encoding='utf-8')
evaluate=(root/'evaluate_visual_graph_fast.py').read_text(encoding='utf-8')
evaluate=evaluate.replace('from fast_vision import BoundaryGraphEncoder','from predictive_nesy.fast_vision import BoundaryGraphEncoder')
evaluate=evaluate.replace('from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel as SubgoalRuleModel,Config',
    'from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel as SubgoalRuleModel,Config,export_program')
evaluate=evaluate.replace('Controller,Program,export_program,successors','Controller,Program,successors')
evaluate=evaluate.replace('anonymous_visual_graph_subgoal_v5','anonymous_visual_memory_subgoal_v6')
(root/'evaluate_visual_memory_subgoal.py').write_text(evaluate,encoding='utf-8')
launch=(root/'start_visual_graph_subgoal.py').read_text(encoding='utf-8')
launch=launch.replace('src_visual_graph_subgoal_v5','src_visual_memory_subgoal_v6').replace('run_visual_graph_subgoal_v5','run_visual_memory_subgoal_v6')
launch=launch.replace('train_visual_graph_subgoal.py','train_visual_memory_subgoal.py')
launch=launch.replace('run_graph_subgoal_v3/checkpoint_000400.pt','run_visual_graph_subgoal_v5/checkpoint_004000.pt')
launch=launch.replace("'6000'","'2000'")
launch=launch.replace("'--eval-every','400']","'--eval-every','400','--lr','1e-4']")
(root/'start_visual_memory_subgoal.py').write_text(launch,encoding='utf-8')
print('Prepared isolated visual-memory entry points')
