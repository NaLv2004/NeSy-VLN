import random
import pytest
import torch
from nesy_vla.symbolic_trust_region import UpdateSnapshot,policy_drift


def test_full_distribution_kl_and_padding_are_counted_exactly():
    p=torch.tensor([[[.5,.5],[.9,.1]],[[.2,.8],[.3,.7]]])
    q=p.clone();q[0,0]=torch.tensor([.75,.25]);q[1,1]=float('nan')
    mask=torch.tensor([[True,True],[True,False]])
    actions=torch.zeros(2,2,dtype=torch.long)
    stats=policy_drift(p,q,actions,mask)
    expected=.5*torch.log(torch.tensor(.5/.75,dtype=torch.float64))+.5*torch.log(torch.tensor(.5/.25,dtype=torch.float64))
    assert torch.allclose(stats,torch.tensor([expected,1,3],dtype=torch.float64),atol=1e-12)
    assert torch.equal(policy_drift(p,p,actions,mask),torch.tensor([0.,0.,3.],dtype=torch.float64))


def test_invalid_candidate_causes_rejection_not_an_unchecked_update():
    p=torch.tensor([[[.5,.5]]]);q=p.clone();q[0,0,0]=float('nan')
    assert torch.isinf(policy_drift(p,q,torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool))[0])
    with pytest.raises(TypeError):policy_drift(p,p,torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool))


def test_rollback_restores_parameters_optimizer_moments_and_rng():
    torch.manual_seed(90);rng=random.Random(90)
    model=torch.nn.Linear(3,2);critic=torch.nn.Linear(3,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.1);value_optimizer=torch.optim.AdamW(critic.parameters(),lr=.1)
    x=torch.ones(2,3)
    for module,opt in [(model,optimizer),(critic,value_optimizer)]:
        module(x).square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    snapshot=UpdateSnapshot(model,optimizer,critic,value_optimizer,rng)
    expected_params=[p.clone() for m in [model,critic] for p in m.parameters()]
    expected_torch=torch.rand(4);expected_python=rng.random()
    for module,opt in [(model,optimizer),(critic,value_optimizer)]:
        module(x).square().sum().backward();opt.step()
    snapshot.restore()
    assert all(torch.equal(a,b) for a,b in zip(expected_params,[p for m in [model,critic] for p in m.parameters()]))
    assert torch.equal(expected_torch,torch.rand(4)) and expected_python==rng.random()
    for opt in [optimizer,value_optimizer]:
        assert all(float(state['step'])==1 for state in opt.state.values())


def test_drift_sums_support_unequal_rank_action_counts():
    p=torch.tensor([[[.5,.5],[.4,.6]],[[.7,.3],[.8,.2]]]);q=p.roll(1,-1)
    mask=torch.tensor([[True,False],[True,True]]);actions=torch.zeros(2,2,dtype=torch.long)
    whole=policy_drift(p,q,actions,mask)
    parts=sum(policy_drift(p[i:i+1],q[i:i+1],actions[i:i+1],mask[i:i+1]) for i in range(2))
    assert torch.allclose(whole,parts,atol=1e-12,rtol=0)
