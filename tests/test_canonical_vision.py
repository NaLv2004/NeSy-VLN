import torch
from torch import nn
from nesy_vla.perception import PredicateGrounder


class TestVision(nn.Module):
    __test__=False
    output_channels=8
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,8,3,padding=1);self.sizes=[]
    def forward(self,x):
        self.sizes.append(len(x));return self.conv(x)


def test_canonical_visual_batches_preserve_tokens_across_input_groupings():
    torch.manual_seed(17);vision=TestVision()
    grounder=PredicateGrounder(d=16,actions=3,vision_backbone=vision,canonical_vision_batch=4)
    images=torch.randn(9,3,16,16)
    whole=grounder.encode(images)
    pieces=torch.cat([grounder.encode(x[None]) for x in images])
    assert vision.sizes==[4]*12
    assert torch.equal(whole,pieces)


def test_canonical_padding_preserves_encoder_and_projection_gradients():
    grounder=PredicateGrounder(d=16,actions=3,vision_backbone=TestVision(),canonical_vision_batch=4)
    images=torch.randn(3,3,16,16,requires_grad=True)
    grounder.encode(images).square().mean().backward()
    assert images.grad is not None and images.grad.abs().sum()>0
    assert grounder.vision.conv.weight.grad.abs().sum()>0
    assert grounder.projection.weight.grad.abs().sum()>0
