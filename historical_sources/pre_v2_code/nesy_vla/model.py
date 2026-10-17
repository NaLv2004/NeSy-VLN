import copy
import torch
from torch import nn
from .generator import VLMProgramGenerator
from .perception import PredicateGrounder
from .logic import SoftLogic
from .runtime import image_tensor


class NavigationModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.generator=VLMProgramGenerator(cfg['model_path'],cfg['program'])
        p=cfg['program']; self.grounder=PredicateGrounder(p['d'],p['a'],cfg['imagenet_weights'])
        self.teacher=copy.deepcopy(self.grounder).requires_grad_(False).eval()
        self.logic=SoftLogic(p['k'],p['a'],p['r'])

    def train(self,mode=True):
        super().train(mode); self.teacher.eval()
        # Disable stochastic depth for exact PPO recomputation; gradients remain enabled.
        self.grounder.vision.eval()
        return self

    @property
    def device(self):return next(self.grounder.parameters()).device

    def step(self,program,image,memory,prior=None,tokens=None):
        if tokens is None:tokens=self.grounder.encode(image_tensor(image,self.cfg['image_size'],self.device))
        observation,confidence=self.grounder.observe(tokens,program.definitions,memory)
        if prior is None:belief=observation; certainty=confidence
        else:
            predicted,prior_c=prior
            belief=(1-confidence)*predicted+confidence*observation
            certainty=1-(1-prior_c)*(1-confidence)
        proof=self.logic.infer(program,belief,certainty)
        value,_=self.grounder.readout(program.definitions,proof['belief'],proof['confidence'])
        return dict(proof,tokens=tokens,observation=observation,observed_confidence=confidence,value=value)

    def advance(self,program,result,action,feedback,memory):
        retention=self.grounder.retain(program.definitions,action)
        b,c,conflict=self.logic.transition(program,result['belief'],result['confidence'],action,retention)
        event=self.grounder.remember(result['tokens'],action,
            torch.tensor(feedback,device=self.device,dtype=torch.float32))
        return (b,c),[*memory[-15:],event],conflict

    @torch.no_grad()
    def update_teacher(self,decay=.995):
        for a,b in zip(self.teacher.parameters(),self.grounder.parameters()):a.lerp_(b,1-decay)

    def transition_target(self,program,image,memory,tokens=None):
        with torch.no_grad():
            if tokens is None:tokens=self.teacher.encode(image_tensor(image,self.cfg['image_size'],self.device))
            return self.teacher.observe(tokens,program.definitions.detach(),[v.detach() for v in memory])
