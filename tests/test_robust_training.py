from types import SimpleNamespace
import torch
from PIL import Image,ImageOps
from nesy_vla.training import train_batch
from nesy_vla.runtime import image_tensor


def exercise(monkeypatch,flags):
    captured=[];encoded=[];augmentation_calls=[]
    def encode(images):
        encoded.append(images.clone());return images.mean((-1,-2))
    def augment(image):
        augmentation_calls.append(image);return ImageOps.invert(image)
    def chunk_loss(model,chunk,expert,normalization,program,tokens,next_tokens,return_loss_terms=False,reference_tokens=None):
        captured.append({'expert':expert,'tokens':tokens.clone(),'reference':reference_tokens})
        return tokens.sum()*0.,{}
    monkeypatch.setattr('nesy_vla.augmentation.perturb_observation',augment)
    monkeypatch.setattr('nesy_vla.training.train_chunk',chunk_loss)
    image=Image.new('RGB',(8,8),(13,78,220))
    chunks=[({'task':'Find a target','images':[image],'feedback':[],'previous':None,
              'rows':[{'image':image,'next_image':image}]},expert) for expert in flags]
    model=SimpleNamespace(cfg={'robust_grounding':True,'image_size':8},device='cpu',
        grounder=SimpleNamespace(encode=encode),teacher=SimpleNamespace(encode=encode),
        generator=SimpleNamespace(generate_many=lambda requests:[None]*len(requests)))
    train_batch(model,chunks)
    expected=image_tensor(image,8,'cpu');inverted=image_tensor(ImageOps.invert(image),8,'cpu')
    for i,flag in enumerate(flags):
        assert torch.equal(encoded[0][i:i+1],inverted if flag else expected)
        assert torch.allclose(captured[i]['tokens'],(inverted if flag else expected).mean((-1,-2)),atol=1e-6,rtol=0)
        if flag:assert torch.allclose(captured[i]['reference'],expected.mean((-1,-2)),atol=1e-6,rtol=0)
        else:assert captured[i]['reference'] is None
    assert len(augmentation_calls)==sum(flags)


def test_mixed_expert_augmentation_preserves_ppo_observations(monkeypatch):
    exercise(monkeypatch,[False,True,False,True])


def test_all_ppo_batch_bypasses_augmentation_and_consistency_loss(monkeypatch):
    exercise(monkeypatch,[False,False])


def test_all_expert_batch_preserves_previous_robust_behavior(monkeypatch):
    exercise(monkeypatch,[True,True])
