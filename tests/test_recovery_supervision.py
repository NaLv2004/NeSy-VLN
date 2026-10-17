from types import SimpleNamespace
import torch
from nesy_vla.training import train_chunk


def test_context_actions_update_history_without_imitation_gradient():
    logits=torch.tensor([[.2,.4,.1],[.4,.1,.3]],requires_grad=True);memories=[];advanced=[]
    model=SimpleNamespace(cfg={'loss':{key:(1. if key=='bc' else 0.) for key in
        ['bc','entropy','transition','conflict','diversity','sparse']}},device='cpu')
    def step(program,image,memory,prior,tokens=None,replay_perceptual_codes=None):
        memories.append(list(memory));zero=logits.sum()*0;observed=zero+torch.ones(4)*.5
        return dict(probs=logits[image].softmax(-1),observation=observed,belief=observed,
            confidence=observed,observed_confidence=observed,tokens=None)
    def advance(program,result,action,feedback,memory):
        advanced.append(action);return (torch.ones(4)*.5,torch.ones(4)),memory+[action],logits.sum()*0
    model.step=step;model.advance=advance
    model.transition_target=lambda *a,**k:(torch.zeros(4),torch.ones(4))
    model.grounder=SimpleNamespace(readout=lambda *a:(None,None))
    model.logic=SimpleNamespace(regularization=lambda program:{'selector_entropy':logits.sum()*0,'rule_weight':logits.sum()*0})
    program=SimpleNamespace(definitions=torch.eye(4))
    rows=[dict(image=0,next_image=1,action=1,feedback=[1.,0.,1.],collision=True,supervision_mask=False),
          dict(image=1,next_image=1,action=2,feedback=[0.,.1,1.],collision=False,supervision_mask=True)]
    loss,stats=train_chunk(model,{'memory':[],'rows':rows},True,program=program)
    torch.testing.assert_close(loss,-logits[1].log_softmax(-1)[2]/2)
    loss.backward();assert torch.equal(logits.grad[0],torch.zeros(3));assert logits.grad[1].abs().sum()>0
    assert advanced==[1,2] and memories==[[],[1]]
    assert stats['expert_action_count']==1 and stats['expert_context_count']==1
