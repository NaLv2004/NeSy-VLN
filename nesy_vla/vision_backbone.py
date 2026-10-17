"""Frozen pretrained visual evidence; no language model or motor-action output."""
import hashlib,json
from collections import OrderedDict
from pathlib import Path
import torch
from torch import nn


class FrozenFeatureCalibration(nn.Module):
    """Training-image channel statistics remove a frozen encoder's common bias.

    No action labels, semantic predicate labels or held-out frames are used.
    This remains entirely inside perception, before Boolean predicate grounding.
    """
    def __init__(self,path,channels):
        super().__init__()
        state=torch.load(path,map_location='cpu',weights_only=True)
        mean,scale=state['mean'].float(),state['scale'].float()
        if mean.shape!=(channels,) or scale.shape!=(channels,):raise ValueError('Calibration channel mismatch')
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('Invalid feature calibration')
        self.register_buffer('mean',mean[None,:,None,None]);self.register_buffer('scale',scale[None,:,None,None])

    def forward(self,features):return (features.float()-self.mean)/self.scale


class FrozenFeatureCache(nn.Module):
    """Bounded RAM cache of immutable image features, shared with the EMA teacher.

    Keys hash the actual normalized pixels. Augmented or otherwise changed
    images cannot silently reuse a clean view's features. No disk cache or map.
    """
    def __init__(self,backbone,entries=4096):
        super().__init__();self.backbone=backbone.requires_grad_(False)
        self.entries=entries;self.cache=OrderedDict();self.hits=0;self.misses=0
    def _apply(self,fn,recurse=True):
        self.cache.clear();return super()._apply(fn,recurse)
    def _load_from_state_dict(self,*args,**kwargs):
        self.cache.clear();return super()._load_from_state_dict(*args,**kwargs)
    def train(self,mode=True):return super().train(False)
    @torch.no_grad()
    def forward(self,image):
        if self.entries<=0:return self.backbone(image)
        cpu=image.detach().float().cpu().contiguous()
        keys=[hashlib.sha256(x.numpy().tobytes()).hexdigest()+str(tuple(image.shape[1:])) for x in cpu]
        needed={}
        for index,key in enumerate(keys):
            if key not in self.cache:needed.setdefault(key,index)
        self.hits+=len(keys)-len(needed);self.misses+=len(needed)
        generated={}
        if needed:
            features=self.backbone(image[list(needed.values())]).detach().cpu()
            generated={key:value.clone() for key,value in zip(needed,features)}
        # Collect before eviction; even a batch larger than cache capacity is valid.
        outputs=[]
        for key in keys:
            value=generated[key] if key in generated else self.cache[key]
            outputs.append(value);self.cache[key]=value;self.cache.move_to_end(key)
        result=torch.stack(outputs).to(image.device)
        while len(self.cache)>self.entries:self.cache.popitem(last=False)
        return result


def qwen_image_patches(image,patch_size=16,merge_size=2,temporal_patch_size=2):
    """Map the existing ImageNet-normalized tensor to official Qwen patch order.

    Geometry is unchanged. The same online pixels are only renormalized and
    packed, with the single photograph repeated along the required time axis.
    """
    batch,channels,height,width=image.shape
    if channels!=3 or height%(patch_size*merge_size) or width%(patch_size*merge_size):
        raise ValueError('Qwen image dimensions must be multiples of patch_size * merge_size')
    mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
    std=image.new_tensor([.229,.224,.225])[None,:,None,None]
    pixels=(image*std+mean-.5)/.5
    gh,gw=height//patch_size,width//patch_size
    patches=pixels[:,None].expand(-1,temporal_patch_size,-1,-1,-1).contiguous()
    patches=patches.view(batch,1,temporal_patch_size,3,gh//merge_size,merge_size,patch_size,
                         gw//merge_size,merge_size,patch_size)
    patches=patches.permute(0,1,4,7,5,8,3,2,6,9).reshape(batch*gh*gw,3*temporal_patch_size*patch_size**2)
    grid=torch.tensor([[1,gh,gw]]*batch,device=image.device,dtype=torch.long)
    return patches,grid


class FrozenQwenVision(nn.Module):
    output_channels=2048
    def __init__(self,model_path,microbatch=8):
        super().__init__()
        from safetensors import safe_open
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        root=Path(model_path);raw=json.loads((root/'config.json').read_text())['vision_config']
        cfg=Qwen3_5VisionConfig(**raw);cfg._attn_implementation='sdpa'
        self.encoder=Qwen3_5VisionModel(cfg).to(dtype=torch.bfloat16)
        index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        selected={k:v for k,v in index.items() if k.startswith('model.visual.')}
        if not selected:raise ValueError('No pretrained visual tensors found')
        state={}
        for filename in sorted(set(selected.values())):
            with safe_open(root/filename,framework='pt',device='cpu') as reader:
                for name,file in selected.items():
                    if file==filename:state[name.removeprefix('model.visual.')]=reader.get_tensor(name)
        self.encoder.load_state_dict(state,strict=True);del state
        self.requires_grad_(False);self.eval();self.microbatch=microbatch
        self.patch_size=cfg.patch_size;self.merge_size=cfg.spatial_merge_size;self.temporal_patch_size=cfg.temporal_patch_size
        self.output_channels=cfg.out_hidden_size

    def train(self,mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self,image):
        answers=[]
        for batch in image.split(self.microbatch):
            pixels,grid=qwen_image_patches(batch,self.patch_size,self.merge_size,self.temporal_patch_size)
            output=self.encoder(pixels.to(self.encoder.dtype),grid_thw=grid,return_dict=True).pooler_output
            height=image.shape[-2]//(self.patch_size*self.merge_size)
            width=image.shape[-1]//(self.patch_size*self.merge_size)
            answers.append(output.reshape(len(batch),height,width,self.output_channels).permute(0,3,1,2).contiguous())
        return torch.cat(answers)
