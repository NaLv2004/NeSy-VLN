"""Verify sparse zero support, corrective gradients, and actual batch backward."""
import json
from pathlib import Path
import torch
from sparse_distribution import probability,loss
from model import FactoredWorld,Config
from data import ROOT,sha
from planner import Program

torch.set_num_threads(3)
scores=torch.tensor([[5.,0.,-5.]],requires_grad=True);labels=torch.tensor([2])
assert probability(scores)[0,2]==0
objective=loss(scores,labels);objective.backward()
assert torch.isfinite(objective) and scores.grad[0,2]<-.99
with torch.no_grad():
    for _ in range(200):
        scores.requires_grad_(True)
        with torch.enable_grad():
            objective=loss(scores,labels);gradient=torch.autograd.grad(objective,scores)[0]
        scores-=.3*gradient
        if probability(scores)[0,2]>.99:break
assert probability(scores)[0,2]>.99
torch.manual_seed(44)
scores=torch.randn(4,6,dtype=torch.float64,requires_grad=True)
labels=torch.tensor([0,2,4,5])
assert torch.autograd.gradcheck(lambda value:loss(value,labels),(scores,),atol=2e-5,rtol=2e-4)
old=ROOT/'run_boundary_balanced_v43'
commit=json.loads((old/'best.COMMITTED.json').read_text());file=old/commit['filename']
assert sha(file)==commit['sha256']
saved=torch.load(file,map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config']));model.load_state_dict(saved['model'])
case=torch.load(ROOT/'run_induced_effects_v39/backward_case_rank0.pt',map_location='cpu',weights_only=False)
data={key:value if key in ('operator_targets','world_updates_encoder') or not torch.is_tensor(value) else value[:8]
      for key,value in case['data'].items()}
data['world_updates_encoder']=True
model.zero_grad(set_to_none=True);objective,metrics=model(data);objective.backward()
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
assert torch.isfinite(objective) and torch.isfinite(norm)
def forbid(*_):raise RuntimeError('Observation grounding invoked during symbolic rule compilation')
handle=model.memory.register_forward_pre_hook(forbid)
with torch.no_grad():
    exported=model.compile(torch.randn(1,model.cfg.context),torch.tensor([0]))
    Program(**exported).validate()
handle.remove()
assert (exported['outcome_probability']==0).any()
record=dict(passed=True,source_commit=commit,zero_probability_target_has_corrective_gradient=True,
    finite_difference_gradient_checked=True,actual_batch_backward_finite=True,
    loss=float(objective.detach()),gradient_norm=float(norm),
    branch_mean_support=float(metrics['branch_mean_supported_observations']),
    excluded_true_outcome_rate=float(metrics['branch_true_outcome_excluded']),
    exact_zero_retained_by_symbolic_export=True,no_runtime_grounding_in_rule_compilation=True,
    navigation_effect='Unproven; contract checks do not measure navigation')
Path(__file__).with_name('sparse_contracts.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
