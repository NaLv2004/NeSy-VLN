import torch
import torch.nn.functional as F
from nesy_vla.losses import transition_loss


def test_inverse_cannot_identify_actions_using_confidence_with_identical_symbols():
    from nesy_vla.losses import inverse_transition_logits
    predictions=torch.tensor([[.95,.05],[.75,.25],[.51,.49]],requires_grad=True)
    target=torch.tensor([1.,0.])
    discrete=inverse_transition_logits(predictions,target,True)
    continuous=inverse_transition_logits(predictions,target,False)
    assert torch.equal(discrete,torch.zeros(3))
    assert continuous[0]>continuous[1]>continuous[2]
    changed=torch.tensor([0.,1.])
    F.cross_entropy(inverse_transition_logits(predictions,changed,True)[None],torch.tensor([1])).backward()
    assert predictions.grad is not None and predictions.grad.abs().sum()>0


def test_inverse_prefers_actual_discrete_transition_match():
    from nesy_vla.losses import inverse_transition_logits
    predictions=torch.tensor([[.9,.1],[.9,.9],[.1,.1]])
    logits=inverse_transition_logits(predictions,torch.tensor([1.,1.]),True)
    assert logits.argmax()==1 and logits[1]>logits[0]>logits[2]


def test_changed_and_stable_groups_have_equal_gradient_budget():
    prediction=torch.full((20,),.25,requires_grad=True)
    source=torch.zeros(20);target=torch.zeros(20);target[0]=1
    loss=transition_loss(prediction,target,source,torch.ones(20),balanced=True)
    gradient=torch.autograd.grad(loss,prediction)[0]
    # Twenty bits: nineteen persistent zeros must not dilute the one actual change.
    assert torch.allclose(loss,.5*(-torch.log(prediction[0])-torch.log(1-prediction[1:]).mean()))
    assert torch.allclose(gradient[0],torch.tensor(-2.))
    assert torch.allclose(gradient[1:].sum(),torch.tensor(2./3.))


def test_empty_change_groups_remain_finite_and_do_not_halve_loss():
    for target in (torch.zeros(5),torch.ones(5)):
        prediction=torch.full((5,),.3,requires_grad=True)
        loss=transition_loss(prediction,target,torch.zeros(5),torch.ones(5),True)
        assert torch.allclose(loss,F.binary_cross_entropy(prediction,target))
        assert torch.isfinite(torch.autograd.grad(loss,prediction)[0]).all()


def test_targets_and_group_assignment_cannot_receive_gradients():
    prediction=torch.tensor([.2,.7],requires_grad=True)
    source=torch.tensor([.1,.9],requires_grad=True)
    target=torch.tensor([1.,1.],requires_grad=True)
    confidence=torch.ones(2,requires_grad=True)
    transition_loss(prediction,target,source,confidence,True).backward()
    assert prediction.grad is not None
    assert source.grad is None and target.grad is None and confidence.grad is None


def test_disabled_option_reproduces_original_loss():
    prediction=torch.tensor([.2,.7]);target=torch.tensor([1.,0.]);confidence=torch.tensor([.1,.8])
    expected=(F.binary_cross_entropy(prediction,target,reduction='none')*confidence.clamp_min(.2)).mean()
    assert torch.allclose(transition_loss(prediction,target,target,confidence,False),expected)
