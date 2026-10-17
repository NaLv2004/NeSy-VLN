from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
for old,new in [('train_graph_subgoal.py','train_visual_graph_subgoal.py'),('evaluate_graph_subgoal.py','evaluate_visual_graph_subgoal.py')]:
    text=(root/old).read_text(encoding='utf-8')
    text=text.replace('predictive_nesy.graph_subgoal_model import GraphSubgoalModel','predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel')
    text=text.replace('predictive_nesy.graph_subgoal_training','predictive_nesy.visual_graph_subgoal_training')
    text=text.replace('anonymous_graph_subgoal_v4','anonymous_visual_graph_subgoal_v5')
    text=text.replace("'evaluate_graph_subgoal.py'","'evaluate_visual_graph_subgoal.py'")
    if old.startswith('train'):
        text=text.replace('training_uses_exact_argmax_transitions=True,','training_uses_exact_argmax_transitions=True,\n        visual_state_labels_ignore_future_actions=True,teacher_forced_dynamics_supervision=True,\n        initialization_scope=\'visual encoder and boundary history; definitions, transitions, subgoal and reconstruction initialized fresh\',')
        old_copy="        if name in current and (old.get('architecture','').startswith('anonymous_graph_subgoal_') or not name.startswith('subgoal.')) and value.shape==current[name].shape:current[name]=value;loaded.append(name)"
        new_copy="        preserve=name=='offsets' or name.startswith(('tokens.','observation.','context_read.','history.','task.','action_embed.','context_norm.'))\n        if preserve and name in current and value.shape==current[name].shape:current[name]=value;loaded.append(name)"
        if old_copy not in text:raise RuntimeError('Initialization template changed')
        text=text.replace(old_copy,new_copy)
    (root/new).write_text(text,encoding='utf-8')
start=(root/'start_graph_subgoal.py').read_text(encoding='utf-8').replace('src_graph_subgoal_v4','src_visual_graph_subgoal_v5').replace('run_graph_subgoal_v4','run_visual_graph_subgoal_v5')
start=start.replace('train_graph_subgoal.py','train_visual_graph_subgoal.py')
(root/'start_visual_graph_subgoal.py').write_text(start,encoding='utf-8')
preflight=(root/'preflight_graph_subgoal.py').read_text(encoding='utf-8')
preflight=preflight.replace('predictive_nesy.graph_subgoal_model import GraphSubgoalModel','predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel as GraphSubgoalModel')
preflight=preflight.replace('from predictive_nesy.graph_subgoal_training import objective,transition_objective,proof_log_support,state_support',
    'from predictive_nesy.visual_graph_subgoal_training import objective,transition_objective\nfrom predictive_nesy.graph_subgoal_training import proof_log_support,state_support')
preflight=preflight.replace("root/'run_anonymous_v5/best_development.pt'","root/'run_graph_subgoal_v3/checkpoint_000400.pt'")
preflight=preflight.replace("    if n in current and not n.startswith('subgoal.') and v.shape==current[n].shape:current[n]=v", "    preserve=n=='offsets' or n.startswith(('tokens.','observation.','context_read.','history.','task.','action_embed.','context_norm.'))\n    if preserve and n in current and v.shape==current[n].shape:current[n]=v")
(root/'preflight_visual_graph_subgoal.py').write_text(preflight,encoding='utf-8')
print('Prepared independent visual-grounded subgoal trainer and evaluator; no running source changed.')
