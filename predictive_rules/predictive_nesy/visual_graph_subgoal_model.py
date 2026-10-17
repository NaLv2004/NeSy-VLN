"""Anonymous states grounded in images under one boundary-generated definition.

Executed-action history may condition the boundary program. Future action tags
cannot manufacture a future-state label without a corresponding visual change.
"""
import torch
from torch.nn import functional as F
from .graph_subgoal_model import GraphSubgoalModel,Config,straight_through


class VisualGraphSubgoalModel(GraphSubgoalModel):
    def __init__(self,config=None):
        super().__init__(config)
        del self.memory_features,self.fuse

    def ground_features(self,program,codes,memory=None):
        features=F.normalize(self.observation(self.embed(codes)),dim=-1)
        definitions=F.normalize(program['definitions'],dim=-1)
        return (8*torch.einsum('bd,bkd->bk',features,definitions).float()).softmax(-1)[:,None]

    def training_targets(self,program,codes,actions):
        # Future RGB is a label source only. The boundary context and generated
        # definitions are fixed, and action tags never enter this label encoder.
        b,t,_=codes.shape
        definitions={'definitions':program['definitions'].repeat_interleave(t,0)}
        return self.ground_features(definitions,codes.flatten(0,1)).reshape(b,t,1,self.cfg.states)
