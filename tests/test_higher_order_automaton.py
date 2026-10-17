import pytest
import torch
from nesy_vla.higher_order_automaton import HigherOrderAutomaton
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton


def case():
    torch.manual_seed(64)
    m=HigherOrderAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True,higher_rules=7)
    x=torch.randint(8,(2,6));goal=torch.tensor([0,1]);p=m.generate(goal,x[:,None],0)
    return m,x,p


def test_higher_clauses_match_explicit_equality_conjunction_sum():
    m,x,p=case();state=torch.rand(2,3,4).softmax(-1)
    actual=m.higher_rule_proof(p,state,0);manual=torch.zeros(2,17)
    for b in range(2):
        weights=[]
        for r in range(7):
            support=p.higher_prior[b,r]
            for f in range(3):
                eq=sum(state[b,f,s]*p.premise[b,r,f,s] for s in range(4))
                support=support*(p.unused[b,r,f]+(1-p.unused[b,r,f])*eq)
            weights.append(support)
        denominator=sum(weights)
        for r in range(7):manual[b]+=weights[r]/denominator*p.higher_conclusions[b,r]
    assert torch.allclose(actual['probabilities'],manual,atol=1e-7,rtol=0)


def test_exact_transition_and_action_proof_and_rule_erasure():
    m,x,p=case();_,previous=m.initial_state(2);actions=torch.full((2,),17)
    r=m.step(p,x,previous,actions,0)
    proof=m.proof(p,r['observed_ids'],previous,actions,r['state_values'],0)
    assert torch.equal(proof['probabilities'],r['probabilities'])
    with torch.no_grad():
        m.conclusions.zero_();m.conclusion_basis.zero_();m.higher_conclusions.zero_();m.higher_conclusion_basis.zero_()
    r=m.step(m.generate(torch.tensor([0,1]),x[:,None],0),x,previous,actions,0)
    assert torch.allclose(r['probabilities'],torch.ones_like(r['probabilities'])/17,atol=1e-7)
    with pytest.raises(TypeError):m.step(p,x.float(),previous,actions,0)


def test_actor_gradient_reaches_learned_higher_premises():
    m,x,p=case();_,memory=m.initial_state(2);a=torch.full((2,),17)
    r=m.step(p,x,memory,a,0);(-r['probabilities'][:,5].log().mean()).backward()
    for param in [m.higher_premise,m.higher_unused,m.higher_conclusions,m.higher_premise_basis,m.higher_mix]:
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0


def test_zero_higher_mix_exactly_recovers_old_pairwise_interpreter():
    m,x,p=case()
    with torch.no_grad():m.higher_mix.fill_(-1000)
    old=ProbabilisticTokenAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True)
    old.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('higher_')})
    _,memory=m.initial_state(2);a=torch.full((2,),17);goal=torch.tensor([0,1])
    assert torch.equal(m.step(m.generate(goal,x[:,None],0),x,memory,a,0)['probabilities'],old.step(old.generate(goal,x[:,None],0),x,memory,a,0)['probabilities'])
