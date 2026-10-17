from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
train=(root/'train_visual_memory_subgoal.py').read_text(encoding='utf-8')
train=train.replace('from predictive_nesy.visual_memory_subgoal_training import objective,transition_objective',
    'from predictive_nesy.rollout_proof_training import objective,transition_objective')
train=train.replace('anonymous_visual_memory_subgoal_v6','anonymous_rollout_proof_subgoal_v8')
train=train.replace('evaluate_visual_memory_subgoal.py','evaluate_rollout_proof.py')
train=train.replace("initialization_scope='full v5 checkpoint plus zero-output visual-history memory; optimizer initialized fresh'",
    "initialization_scope='full v6 step400 model; optimizer initialized fresh'")
train=train.replace('proof_action_supervision=True,learned_joint_state_count=128,',
    "proof_action_supervision=True,learned_joint_state_count=128,\n        proof_state_source='initial state plus exact rules and demonstrated action prefix; no future image premises',\n        exact_amp_transition_lookup=True,")
(root/'train_rollout_proof.py').write_text(train,encoding='utf-8')
evaluate=(root/'evaluate_visual_memory_subgoal.py').read_text(encoding='utf-8')
evaluate=evaluate.replace('anonymous_visual_memory_subgoal_v6','anonymous_rollout_proof_subgoal_v8')
(root/'evaluate_rollout_proof.py').write_text(evaluate,encoding='utf-8')
launch=(root/'start_visual_memory_subgoal.py').read_text(encoding='utf-8')
launch=launch.replace('src_visual_memory_subgoal_v6','src_rollout_proof_v8').replace('run_visual_memory_subgoal_v6','run_rollout_proof_v8')
launch=launch.replace('train_visual_memory_subgoal.py','train_rollout_proof.py')
launch=launch.replace('run_visual_graph_subgoal_v5/checkpoint_004000.pt','run_visual_memory_subgoal_v6/checkpoint_000400.pt')
(root/'start_rollout_proof.py').write_text(launch,encoding='utf-8')
preflight=(root/'preflight_visual_memory_cuda.py').read_text(encoding='utf-8')
preflight=preflight.replace('from predictive_nesy.visual_memory_subgoal_training import objective,transition_objective',
    'from predictive_nesy.rollout_proof_training import objective,transition_objective')
preflight=preflight.replace('run_visual_graph_subgoal_v5/checkpoint_004000.pt','run_visual_memory_subgoal_v6/checkpoint_000400.pt')
preflight=preflight.replace('visual_memory_v6_preflight.json','rollout_proof_v8_preflight.json')
(root/'preflight_rollout_proof.py').write_text(preflight,encoding='utf-8')
print('Prepared rollout-proof training, evaluation and preflight')
