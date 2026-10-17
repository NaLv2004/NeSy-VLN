"""Use the existing audited trainer/evaluator shell with independent model modules."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
text=(root/'evaluate_subgoal.py').read_text()
text=text.replace('predictive_nesy.subgoal_model import SubgoalRuleModel,Config','predictive_nesy.graph_subgoal_model import GraphSubgoalModel as SubgoalRuleModel,Config')
text=text.replace('predictive_nesy.subgoal_interpreter','predictive_nesy.graph_subgoal_interpreter')
text=text.replace("'anonymous_subgoal_rules_v1'","'anonymous_graph_subgoal_v3'")
text=text.replace("planner='exact_subgoal_beam',beam_width=beam_width","planner='complete_finite_state_subgoal_search',state_count=model.cfg.states")
(root/'evaluate_graph_subgoal.py').write_text(text)
text=(root/'train_subgoal.py').read_text()
text=text.replace('predictive_nesy.subgoal_model import SubgoalRuleModel,Config','predictive_nesy.graph_subgoal_model import GraphSubgoalModel as SubgoalRuleModel,Config')
text=text.replace('predictive_nesy.subgoal_training','predictive_nesy.graph_subgoal_training')
text=text.replace('Config(**original[\'model_config\'])','Config()')
text=text.replace("'anonymous_subgoal_rules_v1'","'anonymous_graph_subgoal_v3'")
text=text.replace("['definitions','parents','transitions','subgoal_logits']","['definitions','transitions','subgoal_logits']")
text=text.replace("'discrete conditional-rule beam search; depth10 width128'","'complete finite-state derivation to an anonymous goal; maximum depth10'")
text=text.replace("'evaluate_subgoal.py'","'evaluate_graph_subgoal.py'")
text=text.replace("horizon=32,revision_interval=10,boundary_only_neural_inference=True,no_maps=True,","horizon=32,revision_interval=10,boundary_only_neural_inference=True,no_maps=True,\n        proof_action_supervision=True,learned_joint_state_count=128,")
(root/'train_graph_subgoal.py').write_text(text)
