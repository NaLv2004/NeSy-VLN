import torch
from torch import nn
from torchvision.models import convnext_tiny


def image_grid_encoding(side,d,device=None):
    """Standard two-dimensional sinusoidal positions, with centered unit RMS.

    These are image patch addresses, not navigation predicates or action priors.
    """
    if d%4:raise ValueError('Image position dimension must be divisible by four')
    y,x=torch.meshgrid(torch.arange(side,device=device),torch.arange(side,device=device),indexing='ij')
    frequencies=1./(10000.**(torch.arange(d//4,device=device).float()/(d//4)))
    phases=[axis.flatten().float()[:,None]*frequencies[None] for axis in (x,y)]
    positions=torch.cat([part for angle in phases for part in (angle.sin(),angle.cos())],-1)
    positions=positions-positions.mean(0,keepdim=True)
    return positions/positions.square().mean().sqrt().clamp_min(1e-6)


class EvidencePredicateQuery(nn.Module):
    """Predicate-conditioned visual reading without an additive definition shortcut."""
    def __init__(self,d):
        super().__init__()
        self.attention=nn.MultiheadAttention(d,4,dropout=0.,batch_first=True,bias=False)
        self.query_norm=nn.LayerNorm(d)
        self.evidence_norm=nn.LayerNorm(d,elementwise_affine=False)
        self.read_norm=nn.LayerNorm(d,elementwise_affine=False)

    def forward(self,definitions,evidence):
        query=self.query_norm(definitions);observations=self.evidence_norm(evidence)
        read,_=self.attention(query,observations,observations,need_weights=False)
        # Query specifies the predicate; the value stream contains only actual
        # image/event evidence. No direct action logits are produced here.
        return self.read_norm(read)*query


class PredicateGrounder(nn.Module):
    def __init__(self,d=128,actions=17,weights=None,evidence_only=False,spatial_encoding=False,vision_backbone=None,freeze_vision=False,vision_cache_entries=0,canonical_vision_batch=0,vision_calibration=None):
        super().__init__()
        if vision_backbone is None:
            network=convnext_tiny(weights=None)
            if weights:network.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
            self.vision=network.features;channels=768
        else:self.vision=vision_backbone;channels=vision_backbone.output_channels
        if freeze_vision:
            self.vision.requires_grad_(False)
            if vision_cache_entries:
                from .vision_backbone import FrozenFeatureCache
                self.vision=FrozenFeatureCache(self.vision,vision_cache_entries)
        if vision_calibration:
            if not freeze_vision:raise ValueError('Fixed feature calibration requires a frozen encoder')
            from .vision_backbone import FrozenFeatureCalibration
            self.calibration=FrozenFeatureCalibration(vision_calibration,channels)
        else:self.calibration=nn.Identity()
        self.projection=nn.Linear(channels,d)
        self.position=nn.Parameter(torch.randn(49,d)*.01)
        self.spatial_encoding=spatial_encoding
        self.canonical_vision_batch=int(canonical_vision_batch)
        layer=nn.TransformerDecoderLayer(d,4,4*d,dropout=0.,batch_first=True,norm_first=True)
        self.query=EvidencePredicateQuery(d) if evidence_only else nn.TransformerDecoder(layer,2)
        self.truth=nn.Linear(d,1); self.reliability=nn.Linear(d,1)
        self.action=nn.Embedding(actions,d)
        self.event=nn.Sequential(nn.Linear(2*d+3,d),nn.GELU(),nn.LayerNorm(d))
        self.retention=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1))
        self.outcomes=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,3))
        self.critic=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1))

    def encode(self,image):
        if self.canonical_vision_batch:
            size=self.canonical_vision_batch;result=[]
            for start in range(0,len(image),size):
                part=image[start:start+size];count=len(part)
                if count<size:part=torch.cat([part,part.new_zeros((size-count,*part.shape[1:]))])
                result.append(self._encode(part)[:count])
            return torch.cat(result)
        return self._encode(image)

    def _encode(self,image):
        # BF16 tensor-core convolutions; predicate/logic arithmetic remains FP32.
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=image.is_cuda):
            features=self.calibration(self.vision(image))
            x=torch.nn.functional.adaptive_avg_pool2d(features,(7,7)).flatten(2).transpose(1,2)
            tokens=self.projection(x.to(self.projection.weight.dtype))
        tokens=tokens.float()
        if self.spatial_encoding:
            tokens=torch.nn.functional.layer_norm(tokens,(tokens.shape[-1],))
            tokens=tokens+.5*image_grid_encoding(7,tokens.shape[-1],tokens.device)[None]
        return tokens+self.position[None]

    def observe(self,tokens,definitions,memory):
        evidence=tokens
        if memory:
            evidence=torch.cat([evidence,torch.stack(memory[-16:])[None].to(evidence)],1)
        h=self.query(definitions[None],evidence).squeeze(0)
        return self.truth(h).squeeze(-1).sigmoid(),self.reliability(h).squeeze(-1).sigmoid()

    def retain(self,definitions,action):
        a=self.action.weight[action].expand_as(definitions)
        return self.retention(torch.cat([definitions,a],-1)).squeeze(-1).sigmoid()

    def retain_all(self,definitions):
        n=len(self.action.weight);k=len(definitions)
        d=definitions[None].expand(n,-1,-1)
        a=self.action.weight[:,None].expand(-1,k,-1)
        return self.retention(torch.cat([d,a],-1)).squeeze(-1).sigmoid()

    def readout(self,definitions,belief,confidence,action=None):
        # Auxiliary heads only see the bottleneck; neither can select an action.
        latent=(definitions*(belief*confidence)[:,None]).mean(0)
        global_definition=definitions.mean(0)
        value=self.critic(torch.cat([latent,global_definition],-1)).squeeze(-1)
        outcome=None if action is None else self.outcomes(torch.cat([latent,self.action.weight[action]],-1))
        return value,outcome

    def remember(self,tokens,action,feedback):
        return self.event(torch.cat([tokens.mean((0,1)),self.action.weight[action],feedback],-1))
