"""Learn categorical grounding from visual evidence before patch quantization.

Continuous features are private to neural perception/program generation.
The existing exported program and integer-only hard interpreter are unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from joint_model import JointObservedWorld,Config


class RawObservedWorld(JointObservedWorld):
    def __init__(self,config=None):
        super().__init__(config)
        self.raw_observation=nn.Sequential(nn.LayerNorm(49*128),nn.Linear(49*128,512),nn.GELU(),nn.Linear(512,self.cfg.width))
        self.raw_definition_context=nn.Sequential(nn.LayerNorm(49*128),nn.Linear(49*128,self.cfg.hidden))
        nn.init.zeros_(self.raw_observation[-1].weight);nn.init.zeros_(self.raw_observation[-1].bias)
        nn.init.zeros_(self.raw_definition_context[-1].weight);nn.init.zeros_(self.raw_definition_context[-1].bias)

    def visual_features(self,codes,visual):
        if visual.shape!=(*codes.shape[:-1],49,128):raise ValueError('Missing unquantized current visual evidence')
        with torch.autocast(device_type=codes.device.type,enabled=False):
            return self.observation(self.embed(codes))+self.raw_observation(visual.float().flatten(-2))

    def advance_visual(self,memory,previous,previous_visual,feature,visual):
        # Equality is now measured before 64-way patch quantization. Distinct
        # pictures no longer freeze the memory merely because patch IDs agree.
        changed=visual.ne(previous_visual).flatten(-2).any(-1,keepdim=True)
        feature=torch.where(changed,feature,previous)
        delta=feature-previous
        gain=-torch.expm1(-delta.float().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt())
        updated=self.visual_memory(delta,memory)
        return torch.where(changed,memory+gain.to(memory)*(updated-memory),memory),feature,visual

    def prepare(self,tasks,history,previous_actions,lengths,*,start,boundary_features,boundary_available=None):
        if boundary_available is not None and not bool(boundary_available.all()):raise ValueError('No synthetic/fallback visual evidence')
        p=super().prepare(tasks,history,previous_actions,lengths,start=start,boundary_features=boundary_features,boundary_available=boundary_available)
        with torch.autocast(device_type=history.device.type,enabled=False):
            index=torch.arange(len(tasks),device=history.device)
            context=self.context_norm(p['context_raw'])+self.raw_definition_context(boundary_features[index,lengths-1].float().flatten(-2))
            p['definitions']=self.definitions(context)
            features=self.visual_features(history,boundary_features)
            memory=torch.zeros_like(features[:,0]);previous=features[:,0];visual=boundary_features[:,0]
            for t in range(1,history.shape[1]):
                updated,new_feature,new_visual=self.advance_visual(memory,previous,visual,features[:,t],boundary_features[:,t])
                valid=(t<lengths)[:,None]
                memory=torch.where(valid,updated,memory);previous=torch.where(valid,new_feature,previous)
                visual=torch.where(valid[:,:,None],new_visual,visual)
            p.update(ground_memory=memory,ground_last=previous,ground_codes=history[index,lengths-1],ground_visual=visual)
        return p

    def posterior(self,definitions,feature,memory):
        projected=F.normalize(feature+self.memory_projection(memory),dim=-1)
        return (8*torch.einsum('bd,bkd->bk',projected,F.normalize(definitions,dim=-1)).float()).softmax(-1)

    def ground(self,program,codes):
        if not torch.equal(codes,program['ground_codes']):raise ValueError('Current observation has not been incorporated')
        with torch.autocast(device_type=codes.device.type,enabled=False):
            return self.posterior(program['definitions'],program['ground_last'],program['ground_memory'])[:,None]

    def observe(self,program,codes,visual):
        with torch.autocast(device_type=codes.device.type,enabled=False):
            feature=self.visual_features(codes,visual)
            memory,last,current=self.advance_visual(program['ground_memory'],program['ground_last'],program['ground_visual'],feature,visual)
            program.update(ground_memory=memory,ground_last=last,ground_visual=current,ground_codes=codes)
            return self.ground(program,codes)

    def encode_sequence(self,program,codes,visual=None):
        if visual is None:visual=program.get('_targets_visual')
        if visual is None or visual.shape[:2]!=codes.shape[:2]:raise ValueError('Training sequence requires aligned unquantized images')
        with torch.autocast(device_type=codes.device.type,enabled=False):
            features=self.visual_features(codes,visual)
            memory=program['ground_memory'];previous=program['ground_last'];last_visual=program['ground_visual'];states=[]
            for t in range(codes.shape[1]):
                memory,previous,last_visual=self.advance_visual(memory,previous,last_visual,features[:,t],visual[:,t])
                states.append(self.posterior(program['definitions'],previous,memory))
            return torch.stack(states,1),memory,previous,last_visual


def branch_states(model,p,data):
    with torch.autocast(device_type=data['successors'].device.type,enabled=False):
        before,memory,previous,last_visual=model.encode_sequence(p,data['prefix_codes'],data['prefix_visual'])
        b=len(before);a=model.cfg.actions
        feature=model.visual_features(data['successors'],data['successor_visual']).flatten(0,1)
        memory,feature,_=model.advance_visual(memory.repeat_interleave(a,0),previous.repeat_interleave(a,0),
            last_visual.repeat_interleave(a,0),feature,data['successor_visual'].flatten(0,1))
        after=model.posterior(p['definitions'].repeat_interleave(a,0),feature,memory).view(b,a,model.cfg.states)
        return before[:,-1],after
