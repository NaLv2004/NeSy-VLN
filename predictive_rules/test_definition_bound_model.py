"""Check that rules use generated meanings rather than fixed index semantics."""
import numpy as np
import torch
from predictive_nesy.definition_bound_model import DefinitionBoundModel,Config,export_program
from predictive_nesy.graph_subgoal_interpreter import plan,Program
from predictive_nesy.visual_memory_subgoal_training import objective


def run():
    torch.set_num_threads(2);torch.manual_seed(489)
    c=Config(states=16,decoder_hidden=64);model=DefinitionBoundModel(c).eval()
    codes=torch.randint(0,64,(2,11,49));tasks=torch.tensor([0,1]);past=torch.full((2,1),17);lengths=torch.ones(2,dtype=torch.long)
    p=model.generate(tasks,codes[:,:1],past,lengths,start=0)
    context=model.context_norm(p['context_raw']);perm=torch.randperm(c.states)
    t,g=model.definition_rules(p['definitions'][:,perm],context)
    assert torch.allclose(t,p['transitions'][:,perm,:][:,:,:,perm],atol=2e-7,rtol=1e-5)
    assert torch.allclose(g,p['subgoal_logits'][:,:,perm],atol=2e-6,rtol=1e-5)
    changed=p['definitions'].clone();changed[:,0]+=torch.randn_like(changed[:,0])*.5
    changed_t,changed_g=model.definition_rules(changed,context)
    assert not torch.allclose(p['transitions'],changed_t)
    assert not torch.allclose(p['subgoal_logits'],changed_g)
    # The state renaming must preserve the actual proven actions, not only logits.
    original=export_program(p);initial=model.ground(p,codes[:,0]).argmax(-1)[0].numpy()
    inverse=torch.argsort(perm).numpy();renamed=Program(t[0].detach().numpy(),p['definitions'][0,perm].detach().numpy(),
        np.array([inverse[original.subgoal[0]]]),0).validate()
    route=plan(original,initial);other=plan(renamed,inverse[initial])
    assert route['found']==other['found'] and route['actions']==other['actions']
    # A separate synthetic rule-system case guarantees a nonempty proof; this
    # test goal never enters any navigation, training or executable checkpoint.
    endpoint=int(initial[0])
    for _ in range(10):endpoint=int(original.transitions[endpoint,1].argmax())
    synthetic=Program(original.transitions,original.definitions,np.array([endpoint]),0).validate()
    synthetic_renamed=Program(renamed.transitions,renamed.definitions,np.array([inverse[endpoint]]),0).validate()
    route=plan(synthetic,initial);other=plan(synthetic_renamed,inverse[initial])
    assert route['found'] and route['actions']==other['actions']
    model.train();actions=torch.randint(0,17,(2,10))
    d=dict(task=tasks,start=torch.zeros(2,dtype=torch.long),history=codes,previous_actions=torch.full((2,11),17),
        history_length=lengths,targets=codes,actions=actions,action_valid=torch.ones(2,10,dtype=torch.bool),
        state_valid=torch.ones(2,11,dtype=torch.bool),supervised=torch.ones(2,10,dtype=torch.bool),
        goal_index=torch.full((2,),10),episode_id=['a','b'])
    loss,_=objective(model,d);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.definitions.decoder[-1].weight.grad.abs().sum()>0
    print(dict(passed=True,state_renaming_preserves_rules=True,state_renaming_preserves_goal=True,
        generated_definitions_change_transitions=True,finite_gradients=True,plan_equivalence=True,
        training_started=False))


if __name__=='__main__':run()
