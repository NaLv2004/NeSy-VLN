import torch
import pytest
from nesy_vla.vision_backbone import qwen_image_patches


def test_feature_cache_uses_exact_pixels_and_survives_eviction():
    from torch import nn
    from nesy_vla.vision_backbone import FrozenFeatureCache
    class FakeVision(nn.Module):
        def __init__(self):super().__init__();self.frames=0
        def forward(self,x):self.frames+=len(x);return x*3+2
    backbone=FakeVision();cache=FrozenFeatureCache(backbone,entries=2)
    images=torch.randn(4,3,8,8)
    assert torch.equal(cache(images),backbone(images))
    frames=backbone.frames
    assert torch.equal(cache(images[-2:]),images[-2:]*3+2)
    assert backbone.frames==frames
    modified=images[-1:]+.001;assert torch.equal(cache(modified),modified*3+2)
    assert backbone.frames==frames+1
    cache.to('cpu');assert len(cache.cache)==0


def test_official_qwen_patch_order_and_photometric_normalization():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    torch.manual_seed(31)
    rgb=torch.randint(256,(2,3,224,224),dtype=torch.uint8)
    mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    std=torch.tensor([.229,.224,.225])[None,:,None,None]
    pixels,grid=qwen_image_patches((rgb.float()/255-mean)/std)
    processor=Qwen2VLImageProcessorFast(patch_size=16,temporal_patch_size=2,merge_size=2,
        image_mean=[.5]*3,image_std=[.5]*3)
    expected=processor(images=list(rgb),do_resize=False,return_tensors='pt')
    assert torch.equal(grid,expected['image_grid_thw'])
    assert torch.allclose(pixels,expected['pixel_values'],atol=4e-7)
    with pytest.raises(ValueError):qwen_image_patches(torch.zeros(1,3,225,224))


def test_frozen_visual_tower_shared_without_freezing_learned_grounder(monkeypatch):
    from torch import nn
    from nesy_vla.model import NavigationModel
    class FakeVision(nn.Module):
        output_channels=16
        def __init__(self,*args):
            super().__init__();self.layer=nn.Conv2d(3,16,32,32).requires_grad_(False)
        def forward(self,x):return self.layer(x).to(torch.bfloat16)
    monkeypatch.setattr('nesy_vla.vision_backbone.FrozenQwenVision',FakeVision)
    cfg={'generator_backend':'symbolic_gru','program':dict(k=8,d=32,r=48,a=17,structured=True),
         'imagenet_weights':None,'vision_backbone':'qwen35_frozen','freeze_vision':True,
         'model_path':'mock','evidence_only_grounder':True,'discrete_reasoning':True,'image_size':224}
    model=NavigationModel(cfg)
    assert model.teacher.vision is model.grounder.vision
    assert model.grounder.projection.weight.requires_grad and not model.teacher.projection.weight.requires_grad
    p=model.generator('Find chair.',[],[])
    tokens=model.grounder.encode(torch.randn(1,3,224,224));result=model.step(p,None,[],tokens=tokens)
    loss=-result['probs'][4].log();loss.backward()
    assert model.grounder.projection.weight.grad.norm()>0
    assert all(x.grad is None for x in model.grounder.vision.parameters())
    assert set(result['symbol_codes'].tolist())<={0,1}
    model.update_teacher()
