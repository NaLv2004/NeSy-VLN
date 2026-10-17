"""Action proofs use only the initial state and autonomous rule propagation."""
from .visual_graph_subgoal_training import objective as common_objective
from .visual_memory_subgoal_training import transition_objective


def objective(model,batch):
    return common_objective(model,batch,proof_from_predicted_state=True)
