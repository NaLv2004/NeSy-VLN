import torch
import pytest
from nesy_vla.vision_backbone import FrozenFeatureCalibration

def test_feature_calibration_removes_training_bias_and_preserves_spatial_contrasts(tmp_path):
    mean=torch.tensor([20.,-30.,100.]);scale=torch.tensor([2.,4.,5.])
    path=tmp_path/'training_statistics.pt';torch.save(dict(mean=mean,scale=scale),path)
    module=FrozenFeatureCalibration(path,3)
    signal=torch.randn(4,3,7,7,requires_grad=True)
    features=signal*scale[None,:,None,None]+mean[None,:,None,None]
    calibrated=module(features)
    assert torch.allclose(calibrated,signal,atol=2e-6)
    calibrated.square().mean().backward();assert signal.grad.norm()>0
    assert not list(module.parameters())
    with pytest.raises(ValueError):FrozenFeatureCalibration(path,4)

def test_reject_invalid_feature_statistics(tmp_path):
    path=tmp_path/'bad.pt';torch.save(dict(mean=torch.zeros(2),scale=torch.tensor([1.,0.])),path)
    with pytest.raises(ValueError):FrozenFeatureCalibration(path,2)
