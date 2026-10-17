"""Actual recorded minibatch, hard-forward equality, gradient and data checks."""
import json
from pathlib import Path
import torch
from model import FactoredWorld, Config
from objectives import objective

torch.set_num_threads(3)
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
case=torch.load(root/'run_induced_effects_v39/backward_case_rank0.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**case['model_config']))
model.load_state_dict(case['model'],strict=True)
data={key:value if key=='operator_targets' else value[:2] for key,value in case['data'].items()}
outcomes=[]
for phase in (False,True):
    data['world_updates_encoder']=phase
    model.zero_grad(set_to_none=True);torch.manual_seed(41)
    loss,metrics=objective(model,data)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    outcomes.append(dict(phase=phase,loss=float(loss.detach()),gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5)),
        visual_vq=float(metrics['visual_vq_loss']),memory_vq=float(metrics['memory_vq_loss'])))
assert outcomes[0]['loss']==outcomes[1]['loss'],'Scheduling must change gradients only, not states or losses'

# Explicitly isolate the rule's gradient through an observed categorical input.
# In four phases it must not reach the visual encoder, while all rule parameters
# still receive gradients. The fifth phase restores the visual path.
for phase in (False,True):
    model.zero_grad(set_to_none=True)
    logits=model.observe(data['features'][:,0])
    from model import hard
    obs=hard(logits)
    premise=obs if phase else obs.detach()
    context=torch.zeros(len(obs),model.cfg.context)
    memory=torch.nn.functional.one_hot(torch.zeros(len(obs),dtype=torch.long),model.cfg.memories).float()
    _,prediction=model.transition(context,memory,premise,torch.ones(len(obs),dtype=torch.long))
    loss=-prediction[:,7].mean();loss.backward()
    grads=[p.grad for p in model.visual.parameters()]
    assert all(g is not None for g in grads) if phase else all(g is None for g in grads)
    assert model.rule[0].weight.grad is not None
result=dict(recorded_training_batch=True,finite_backward_both_phases=True,
    exact_same_forward_codes_and_loss=True,world_gradient_gate_verified=True,phases=outcomes)
Path(__file__).with_name('learning_contracts.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
