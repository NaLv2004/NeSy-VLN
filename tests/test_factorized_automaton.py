import pytest
import torch
from nesy_vla.factorized_automaton import FactorizedTokenAutomaton,categorical_usage_loss


def example(reset=False):
    torch.manual_seed(51)
    m=FactorizedTokenAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,reset_memory=reset)
    x=torch.randint(8,(2,6));task=torch.tensor([0,1]);program=m.generate(task,x[:,None],0)
    previous,values=m.initial_state(2);action=torch.tensor([17,3])
    result=m.step(program,x,values,action,0)
    return m,x,program,previous,values,action,result


def test_discrete_proof_replays_transition_and_entire_action_distribution():
    for reset in [False,True]:
        m,x,p,previous,values,action,r=example(reset)
        proof=m.proof(p,r['observed_ids'],previous,action,r['state_ids'],0)
        assert torch.equal(proof['probabilities'],r['probabilities'])
        assert torch.equal(proof['transition_probabilities'],r['state_probabilities'])
        assert torch.allclose(r['probabilities'].sum(-1),torch.ones(2))


def test_no_visual_or_soft_state_bypass_and_program_expiry():
    m,x,p,previous,values,action,r=example()
    with pytest.raises(TypeError):m.step(p,x.float(),values,action,0)
    with pytest.raises(ValueError):m.apply_action_rules(p,torch.ones_like(values)/m.states,0)
    with pytest.raises(ValueError):m.step(p,x,values,action,10)
    with pytest.raises(ValueError):m.generate(torch.tensor([0,1]),x[:,None],1)
    with pytest.raises(ValueError):m.generate(torch.tensor([0,1]),x[:,None].repeat(1,2,1),0)
    bad=(r['state_ids']+1)%m.states
    with pytest.raises(ValueError):m.proof(p,r['observed_ids'],previous,action,bad,0)


def test_supervision_reaches_learned_definitions_transitions_and_conclusions():
    m,x,p,previous,values,action,r=example()
    loss=-r['probabilities'][:,0].log().mean()+.2*torch.nn.functional.cross_entropy(r['reconstruction'].flatten(0,1),x.flatten())
    loss.backward()
    for param in [m.definitions,m.transition,m.conclusions,m.context.weight,m.tokens.weight]:
        assert param.grad is not None and param.grad.norm()>0
    assert m.predict[-1].weight.grad is None


def test_auxiliary_heads_cannot_change_actions_and_rule_erasure_changes_them():
    m,x,p,previous,values,action,r=example()
    before=r['probabilities'].detach().clone()
    with torch.no_grad():
        for module in [m.predict,m.inverse,m.reconstruct]:
            for parameter in module.parameters():parameter.normal_(100,10)
    assert torch.equal(before,m.step(p,x,values,action,0)['probabilities'])
    with torch.no_grad():m.conclusions.zero_();m.conclusion_basis.zero_()
    p=m.generate(torch.tensor([0,1]),x[:,None],0)
    erased=m.step(p,x,values,action,0)['probabilities']
    assert torch.allclose(erased,torch.ones_like(erased)/m.actions)
    assert not torch.allclose(before,erased)


def test_ten_action_revision_changes_definitions_and_rule_conclusions():
    m,x,p,previous,values,action,r=example()
    other=m.generate(torch.tensor([0,1]),(7-x)[:,None].repeat(1,11,1),10)
    assert not torch.equal(p.definitions,other.definitions)
    assert not torch.equal(p.transition_delta,other.transition_delta)
    assert not torch.equal(p.action_delta,other.action_delta)
    m.step(other,x,values,action,10)


def test_hard_usage_detects_collapsed_ids_despite_nearly_uniform_soft_scores():
    logits=torch.zeros(8,3,4,requires_grad=True)
    with torch.no_grad():logits[:,:,0]=.0001
    p=logits.softmax(-1);valid=torch.ones(8,dtype=torch.bool)
    soft=categorical_usage_loss(p,valid);hard=categorical_usage_loss(p,valid,True)
    assert abs(float(soft.detach()))<1e-6
    assert torch.allclose(hard,torch.tensor(4.).log())
    hard.backward();assert torch.isfinite(logits.grad).all() and logits.grad.norm()>0


def test_hard_usage_ignores_padding_and_is_invariant_to_symbol_id_permutation():
    p=torch.eye(4)[:,None,:].repeat(1,2,1);valid=torch.ones(4,dtype=torch.bool)
    balanced=categorical_usage_loss(p,valid,True);assert balanced==0
    padded=torch.cat([p,torch.tensor([[[1.,0,0,0]]]).repeat(5,2,1)])
    mask=torch.cat([valid,torch.zeros(5,dtype=torch.bool)])
    assert categorical_usage_loss(padded,mask,True)==balanced
    assert categorical_usage_loss(padded[:,:,[3,1,0,2]],mask,True)==balanced


@pytest.mark.parametrize('banks',[False,True])
def test_goal_bridge_and_program_banks_preserve_source_actions_before_training(banks):
    m,x,p,previous,values,action,r=example()
    improved=FactorizedTokenAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=banks,context_bridge=True)
    improved.initialize_from_shared(m.state_dict())
    program=improved.generate(torch.tensor([0,1]),x[:,None],0)
    result=improved.step(program,x,values,action,0)
    assert torch.equal(result['probabilities'],r['probabilities'])
    assert torch.equal(result['observed_ids'],r['observed_ids'])
    proof=improved.proof(program,result['observed_ids'],previous,action,result['state_ids'],0)
    assert torch.equal(proof['probabilities'],result['probabilities'])
    (-result['probabilities'][:,0].log().mean()).backward()
    assert improved.bridge.weight.grad.norm()>0


def test_goal_program_banks_are_learned_rule_conclusions_and_require_goal_token():
    m=FactorizedTokenAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True)
    x=torch.randint(8,(1,6)).repeat(2,1);tasks=torch.tensor([0,1]);_,prior=m.initial_state(2)
    with torch.no_grad():
        m.conclusions.zero_();m.conclusion_basis.zero_();m.conclusions[0,:,:,:,5]=20;m.conclusions[1,:,:,:,9]=20
    p=m.generate(tasks,x[:,None],0);r=m.step(p,x,prior,torch.full((2,),17,dtype=torch.long),0)
    assert r['probabilities'].argmax(-1).tolist()==[5,9]
    with pytest.raises(ValueError):m.program_from_coefficients(p.coefficients,0)
    proof=m.proof(p,r['observed_ids'],torch.zeros(2,3,dtype=torch.long),torch.full((2,),17,dtype=torch.long),r['state_ids'],0)
    assert torch.equal(proof['probabilities'],r['probabilities'])
