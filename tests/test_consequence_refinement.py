import torch
import pytest
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.consequence_refinement import refine_automaton,merge_child_belief,ConsequenceDecoder,balanced_consequence_loss,discrete_memory_equalities

@pytest.mark.parametrize('task_banks',[False,True])
def test_state_refinement_preserves_policy_over_causal_sequence(task_banks):
    torch.manual_seed(19)
    p=ProbabilisticTokenAutomaton(patches=5,vocabulary=7,factors=3,states=4,width=8,rank=2,task_banks=task_banks,context_bridge=True)
    c,arch=refine_automaton(p);assert arch['states']==8
    x=torch.randint(7,(3,12,5));task=torch.tensor([0,1,0]);a=torch.full((3,),17,dtype=torch.long)
    _,pb=p.initial_state(3);_,cb=c.initial_state(3)
    for step in range(12):
        if step%10==0:
            pp=p.generate(task,x[:,max(0,step-10):step+1],step);cp=c.generate(task,x[:,max(0,step-10):step+1],step)
        pr=p.step(pp,x[:,step],pb,a,step);cr=c.step(cp,x[:,step],cb,a,step)
        torch.testing.assert_close(pr['probabilities'],cr['probabilities'],atol=2e-6,rtol=0)
        torch.testing.assert_close(pr['state_values'],merge_child_belief(cr['state_values']),atol=2e-6,rtol=0)
        assert torch.equal(pr['observed_ids'],cr['observed_ids']//2)
        proof=c.proof(cp,cr['observed_ids'],cb,a,cr['state_values'],step)
        torch.testing.assert_close(proof['probabilities'],cr['probabilities'],atol=1e-7,rtol=0)
        torch.testing.assert_close(p.next_observation(pr['state_values'],a.clamp_max(16)),c.next_observation(cr['state_values'],a.clamp_max(16)),atol=2e-6,rtol=0)
        pb=pr['state_values'];cb=cr['state_values'];a=torch.randint(17,(3,))

def test_consequence_decoder_rejects_continuous_bypass():
    head=ConsequenceDecoder(4,8)
    with pytest.raises(ValueError):head(torch.full((2,4,8),.125),torch.tensor([0,1]))

def test_opposite_execution_labels_drive_different_grounding_gradients():
    torch.manual_seed(4)
    logits=torch.zeros(2,4,8,requires_grad=True)
    probability=logits.softmax(-1);hard=torch.nn.functional.one_hot(probability.argmax(-1),8).float()
    symbols=hard+probability-probability.detach()
    head=ConsequenceDecoder(4,8)
    labels=torch.stack([torch.zeros(16,2),torch.ones(16,2)])
    loss=balanced_consequence_loss(head(symbols,torch.tensor([0,0])),labels);loss.backward()
    assert logits.grad.abs().sum()>0
    assert not torch.allclose(logits.grad[0],logits.grad[1])

def test_memory_auxiliary_breaks_duplicated_transition_conclusion_symmetry():
    torch.manual_seed(18)
    parent=ProbabilisticTokenAutomaton(patches=5,vocabulary=7,factors=4,states=4,width=8,rank=2,task_banks=True)
    model,_=refine_automaton(parent);head=ConsequenceDecoder(4,8)
    x=torch.randint(7,(3,5));tasks=torch.tensor([0,1,0]);_,prior=model.initial_state(3)
    program=model.generate(tasks,x[:,None],0)
    result=model.step(program,x,prior,torch.full((3,),17,dtype=torch.long),0)
    pairs=model.transition.detach()[...,::2]-model.transition.detach()[...,1::2]
    assert pairs.abs().max()==0
    symbols=discrete_memory_equalities(result['state_values'])
    assert torch.equal(symbols.detach(),symbols.detach().round())
    labels=torch.randint(2,(3,16,2)).float()
    balanced_consequence_loss(head(symbols,tasks),labels).backward()
    gradient=model.transition.grad
    assert gradient is not None
    assert (gradient[...,::2]-gradient[...,1::2]).abs().max()>1e-8

def test_discrete_memory_auxiliary_rejects_unnormalized_features():
    with pytest.raises(ValueError):discrete_memory_equalities(torch.ones(2,4,8))
