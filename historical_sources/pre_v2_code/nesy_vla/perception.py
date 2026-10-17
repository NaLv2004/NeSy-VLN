import torch
from torch import nn
from torchvision.models import convnext_tiny


class PredicateGrounder(nn.Module):
    def __init__(self,d=128,actions=17,weights=None):
        super().__init__()
        network=convnext_tiny(weights=None)
        if weights:
            network.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        self.vision=network.features
        self.projection=nn.Linear(768,d)
        self.position=nn.Parameter(torch.randn(49,d)*.01)
        layer=nn.TransformerDecoderLayer(d,4,4*d,dropout=0.,batch_first=True,norm_first=True)
        self.query=nn.TransformerDecoder(layer,2)
        self.truth=nn.Linear(d,1); self.reliability=nn.Linear(d,1)
        self.action=nn.Embedding(actions,d)
        self.event=nn.Sequential(nn.Linear(2*d+3,d),nn.GELU(),nn.LayerNorm(d))
        self.retention=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1))
        self.outcomes=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,3))
        self.critic=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1))

    def encode(self,image):
        # BF16 tensor-core convolutions; predicate/logic arithmetic remains FP32.
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=image.is_cuda):
            features=self.vision(image)
            x=torch.nn.functional.adaptive_avg_pool2d(features,(7,7)).flatten(2).transpose(1,2)
            tokens=self.projection(x)
        return tokens.float()+self.position[None]

    def observe(self,tokens,definitions,memory):
        evidence=tokens
        if memory:
            evidence=torch.cat([evidence,torch.stack(memory[-16:])[None].to(evidence)],1)
        h=self.query(definitions[None],evidence).squeeze(0)
        return self.truth(h).squeeze(-1).sigmoid(),self.reliability(h).squeeze(-1).sigmoid()

    def retain(self,definitions,action):
        a=self.action.weight[action].expand_as(definitions)
        return self.retention(torch.cat([definitions,a],-1)).squeeze(-1).sigmoid()

    def readout(self,definitions,belief,confidence,action=None):
        # Auxiliary heads only see the bottleneck; neither can select an action.
        latent=(definitions*(belief*confidence)[:,None]).mean(0)
        global_definition=definitions.mean(0)
        value=self.critic(torch.cat([latent,global_definition],-1)).squeeze(-1)
        outcome=None if action is None else self.outcomes(torch.cat([latent,self.action.weight[action]],-1))
        return value,outcome

    def remember(self,tokens,action,feedback):
        return self.event(torch.cat([tokens.mean((0,1)),self.action.weight[action],feedback],-1))
