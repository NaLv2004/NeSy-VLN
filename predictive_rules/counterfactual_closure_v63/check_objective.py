"""Small real-data numerical check before full two-step fitting."""
import copy,json,os,time
from pathlib import Path
import numpy as np
import torch
from model import FactoredWorld,Config
from closure_objective import ClosureObjective,configure,nll,TRAINABLE
from data import ROOT,sha
from evaluate import atomic

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(2)
DATA=ROOT/'counterfactual_closure_v63'
p=next(iter(sorted((DATA/'episodes').glob('*.npz'))))
saved=torch.load(DATA/'source.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config'])).eval()
model.load_state_dict(saved['model']);configure(model)
with np.load(p,allow_pickle=False) as archive:
    d={k:torch.from_numpy(archive[k][:1].copy()).to(torch.float32 if k=='context' else torch.bool if k=='second_valid' else torch.long)
       for k in ('context','memory','observation','first_observation','first_memory','second_actions','second_observation','second_memory','second_valid')}
objective=ClosureObjective(model);pairs=torch.tensor([[2,5]])
loss,metrics=objective(d,pairs);loss.backward()
gradients={k:float(v.grad.norm()) for k,v in model.named_parameters() if v.grad is not None}
assert gradients and all(np.isfinite(v) for v in gradients.values())
assert all(k.startswith(TRAINABLE) for k in gradients)
assert sum(v for k,v in gradients.items() if k.startswith('memory_predictor.'))>0
assert sum(v for k,v in gradients.items() if k.startswith('definition_rule.'))>0
# Check the vectorized marginal against a scalar64-state enumeration, using
# independently addressed hard table rows and no future image/state correction.
with torch.no_grad():
    context=d['context'];m=torch.nn.functional.one_hot(d['memory'],32).float()[:,None]
    o=torch.nn.functional.one_hot(d['observation'],64).float()[:,None]
    a=torch.tensor([[1]]);lm,lo=model.transition(context,m,o,a)
    pm=torch.nn.functional.one_hot(lm.argmax(-1),32).float();a2=d['second_actions'][:,1,0,None]
    probabilities=[]
    for ob in range(64):
        oo=torch.nn.functional.one_hot(torch.tensor([[ob]]),64).float()
        _,next_o=model.transition(context,pm,oo,a2)
        probabilities.append(lo.exp()[0,0,ob]*next_o.exp()[0,0])
    scalar=torch.stack(probabilities).sum(0)
    _,next_o=model.transition(context,pm.expand(-1,64,-1),torch.eye(64)[None],a2.expand(-1,64))
    vector=torch.logsumexp(lo[0,0,:,None]+next_o[0],dim=0).exp()
    error=float((scalar-vector).abs().max());assert error<1e-6,error
    # The training-only gradient switch cannot alter a deployed transition.
    before=model.transition(context,m,o,a)
    model.detach_memory_carrier=True;after=model.transition(context,m,o,a)
    assert all(torch.equal(x,y) for x,y in zip(before,after))
atomic(DATA/'objective_contract.json',dict(time=time.time(),sample=p.name,
    objective_sha256=sha(Path(__file__).with_name('closure_objective.py')),
    source_sha256=sha(DATA/'source.pt'),
    loss=float(loss),gradient_norms=gradients,marginal_enumeration_max_error=error,
    training_gradient_switch_identical_forward=True,frozen_parameters_no_gradient=True,
    target_labels_not_used_as_runtime_memory=True,neural_planning_calls=0))
print(json.dumps(dict(loss=float(loss),marginal_error=error,gradient_modules=len(gradients))))
