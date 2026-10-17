"""CPU gradient audit of the observation/rule learning separation."""
import json,random,time
import torch
from adaptive_model import AdaptiveObservedWorld,Config
from train import ROOT,Windows,prepare,observed_objective
from predictive_nesy.subgoal_data import batch
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures

torch.set_num_threads(2);torch.manual_seed(27)
saved=torch.load(ROOT/'run_state512_v25/checkpoint_003000.pt',map_location='cpu',weights_only=False)
model=AdaptiveObservedWorld(Config(**saved['model_config']))
missing,unexpected=model.load_state_dict(saved['model'],strict=False)
assert not unexpected and all(n.startswith(('correction.','revision.')) for n in missing)
corpus=Windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
features=RawBoundaryFeatures(ROOT/'prequant_visual_features_v18','cpu')
rows=[corpus.window(*key) for key in random.Random(27).choices(corpus.by_start[7],k=4)]
data=batch(rows,'cpu');p=prepare(model,data,features,rows)
loss,metrics,labels,parts=observed_objective(model,data,p,True)
parameters=list(model.correction.parameters())
leak=torch.autograd.grad(parts['dynamics']+parts['proof'],parameters,allow_unused=True,retain_graph=True)
assert all(g is None or float(g.abs().max())==0 for g in leak)
grad=torch.autograd.grad(parts['correction'],parameters,allow_unused=True,retain_graph=True)
norm=sum(float(g.norm()) for g in grad if g is not None);assert norm>0
logit=torch.zeros(4,requires_grad=True);net_benefit=torch.tensor([2.,-2.,1.,-1.]);target=(net_benefit>0).float()
gate_loss=(-logit.sigmoid()*net_benefit+.25*torch.nn.functional.binary_cross_entropy_with_logits(logit,target,reduction='none')*net_benefit.abs().clamp(.2,4.)).mean()
gradient=torch.autograd.grad(gate_loss,logit)[0]
assert (gradient[net_benefit>0]<0).all() and (gradient[net_benefit<0]>0).all()
with torch.no_grad():
    observed=labels[:,0]
    before=model.correction.selected(p['definitions'],p['context_raw'],observed,observed).argmax(-1)
assert torch.equal(before,observed)
print(json.dumps(dict(passed=True,dynamics_and_proof_gradient_to_correction=False,correction_supervision_gradient_norm=norm,
                     gate_gradient_cost_direction_correct=True,initial_observation_identity=True,metrics=metrics)),flush=True)
