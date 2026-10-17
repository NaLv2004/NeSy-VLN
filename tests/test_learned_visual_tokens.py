import pytest
import torch
from nesy_vla.learned_visual_tokens import SymbolTokens,LearnedVisualAlphabet,TrainableAlphabetAutomaton


def case():
    torch.manual_seed(65)
    q=LearnedVisualAlphabet(torch.randn(8,12));m=TrainableAlphabetAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True)
    x=torch.randn(2,6,12);task=torch.tensor([0,1]);_,prior=m.initial_state(2);action=torch.full((2,),17)
    return q,m,x,task,prior,action


def test_training_forward_and_deployed_integer_program_are_exactly_equal():
    q,m,x,task,prior,action=case();tokens=q(x)
    p=m.generate(task,tokens[:,None],0);r=m.step(p,tokens,prior,action,0)
    plain=m.step(m.generate(task,tokens.ids[:,None],0),tokens.ids,prior,action,0)
    assert torch.equal(r['probabilities'],plain['probabilities'])
    assert torch.equal(p.coefficients,m.generate(task,tokens.ids[:,None],0).coefficients)
    proof=m.proof(p,r['observed_ids'],prior,action,r['state_values'],0)
    assert torch.equal(proof['probabilities'],r['probabilities'])


def test_action_loss_trains_alphabet_only_through_hard_symbol_boundary():
    q,m,x,task,prior,action=case();t=q(x);r=m.step(m.generate(task,t[:,None],0),t,prior,action,0)
    (-r['probabilities'][:,5].log().mean()).backward()
    for parameter in [q.centers,q.projection_delta,m.definitions,m.transition,m.conclusions,m.tokens.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm()>0


def test_continuous_features_and_soft_noninteger_relaxations_rejected():
    q,m,x,task,prior,action=case();t=q(x)
    with pytest.raises(ValueError):m.generate(task,x[:,None],0)
    with pytest.raises(TypeError):m.generate(task,x[:,:,0][:,None],0)
    with pytest.raises(ValueError):m.generate(task,SymbolTokens(t.ids[:,None],torch.ones_like(t.values[:,None])/8),0)
    with pytest.raises(ValueError):q(torch.randn(2,6,5))


def test_preserving_token_ids_preserves_all_decisions():
    q,m,x,task,prior,action=case();a=q(x);z=q(x+1e-6)
    assert torch.equal(a.ids,z.ids)
    for t in [a,z]:
        r=m.step(m.generate(task,t[:,None],0),t,prior,action,0)
        if t is a:before=r['probabilities'].detach()
        else:assert torch.equal(before,r['probabilities'])
