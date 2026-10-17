import torch
from nesy_vla.learned_visual_tokens import LearnedVisualAlphabet,TrainableAlphabetAutomaton

def test_pixels_train_encoder_but_deployed_decision_depends_only_on_discrete_ids():
    torch.manual_seed(72)
    encoder=torch.nn.Conv2d(3,12,1)
    alphabet=LearnedVisualAlphabet(torch.randn(8,12)).requires_grad_(False)
    model=TrainableAlphabetAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True)
    pixels=torch.randn(2,3,2,3);features=encoder(pixels).flatten(2).transpose(1,2);tokens=alphabet(features)
    goal=torch.tensor([0,1]);_,prior=model.initial_state(2);action=torch.full((2,),17)
    program=model.generate(goal,tokens[:,None],0);result=model.step(program,tokens,prior,action,0)
    (-result['probabilities'][:,5].log().mean()).backward()
    assert encoder.weight.grad.norm()>0 and torch.isfinite(encoder.weight.grad).all()
    # Throw away image, feature tensors and the surrogate categorical object.
    ids=tokens.ids.clone();deployed=model.step(model.generate(goal,ids[:,None],0),ids,prior,action,0)
    assert torch.equal(deployed['probabilities'],result['probabilities'])
    proof=model.proof(program,result['observed_ids'],prior,action,result['state_values'],0)
    assert torch.equal(proof['probabilities'],result['probabilities'])
