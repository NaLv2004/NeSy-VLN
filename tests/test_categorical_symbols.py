import pytest
import torch
from nesy_vla.symbols import tokenize_categories


def test_category_facts_are_hard_with_competitive_gradients():
    raw=torch.tensor([[.1,.9,.2,.3, 3.,2.,1.,0.]],requires_grad=True)
    state=tokenize_categories(raw,2);state.validate()
    assert state.codes.tolist()==[[0,1,0,0,1,0,0,0]]
    assert state.values.reshape(1,2,4).sum(-1).tolist()==[[1.,1.]]
    (state.values*torch.arange(8)).sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
    torch.testing.assert_close(raw.grad.reshape(1,2,4).sum(-1),torch.zeros(1,2),atol=1e-6,rtol=0)


def test_alphabet_shape_and_nonfinite_inputs_are_rejected():
    for count in [0,3,8,True]:
        with pytest.raises(ValueError):tokenize_categories(torch.zeros(8),count)
    with pytest.raises(FloatingPointError):tokenize_categories(torch.full((8,),float('nan')),2)


def test_groupwise_shift_does_not_change_tokens_or_gradients():
    a=torch.randn(3,8,requires_grad=True);b=(a.detach()+torch.tensor([7.]*4+[-3.]*4)).requires_grad_()
    sa=tokenize_categories(a,2);sb=tokenize_categories(b,2)
    assert torch.equal(sa.codes,sb.codes)
    weights=torch.randn_like(a);(sa.values*weights).sum().backward();(sb.values*weights).sum().backward()
    torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)
