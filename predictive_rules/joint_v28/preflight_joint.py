"""Check real-data joint gradients and the hard-state inverse boundary on CPU."""
import copy,json,random
import torch
from joint_model import JointObservedWorld,Config,representation_loss,branch_representation_loss
from train import ROOT,Windows,prepare,observed_objective
from predictive_nesy.subgoal_data import batch,TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures

torch.set_num_threads(2);torch.manual_seed(28);rng=random.Random(28)
saved=torch.load(ROOT/'run_state512_v25/checkpoint_003000.pt',map_location='cpu',weights_only=False)
model=JointObservedWorld(Config(**saved['model_config']))
missing,unexpected=model.load_state_dict(saved['model'],strict=False)
assert not unexpected and all(n.startswith(('inverse_discrete.','correction.','revision.')) for n in missing)
teacher=copy.deepcopy(model).eval().requires_grad_(False)
corpus=Windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cpu')
rows=[corpus.window(*key) for key in rng.choices(corpus.by_start[7],k=4)];data=batch(rows,'cpu')
p=prepare(model,data,features,rows)
with torch.no_grad():
    tp=prepare(teacher,data,features,rows);labels=teacher.training_targets(tp,data['targets'],data['actions']).argmax(-1).squeeze(-1)
loss,metrics,_=observed_objective(model,data,p,target_labels=labels)
aux,extra=representation_loss(model,p,data,labels)
(loss+aux).backward()
def gradients():
    return {key:sum(float(v.grad.norm()) for n,v in model.named_parameters() if n.startswith(key) and v.grad is not None)
            for key in ['observation.','tokens.','definitions.','visual_memory.','inverse_discrete.','definition_rules.']}
demo_gradients=gradients();assert all(v>0 for v in demo_gradients.values()),demo_gradients
model.zero_grad(set_to_none=True)
bank=TransitionBank(saved['manifest']['transition_banks'][0]['path'],saved['manifest']['data']['train_layouts'])
branches=bank.sample(rng,4,'cpu');raw,available=features.transitions(branches,bank.directory)
lengths=torch.full((4,),branches['history'].shape[1],dtype=torch.long)
kwargs=dict(start=branches['start'],boundary_features=raw,boundary_available=available)
p=model.prepare(branches['task'],branches['history'],branches['previous_actions'],lengths,**kwargs)
with torch.no_grad():tp=teacher.prepare(branches['task'],branches['history'],branches['previous_actions'],lengths,**kwargs)
branch_loss,bm=branch_representation_loss(model,teacher,p,tp,branches);branch_loss.backward();branch_gradients=gradients()
assert all(v>0 for v in branch_gradients.values()),branch_gradients
try:model.inverse_from_discrete(torch.full((1,512),1/512),torch.full((1,512),1/512))
except ValueError:pass
else:raise AssertionError('Soft inverse input accepted')
assert all(p.grad is None for p in teacher.parameters())
assert torch.isfinite(branch_loss) and torch.isfinite(loss+aux)
print(json.dumps(dict(passed=True,demonstration_gradients=demo_gradients,branch_gradients=branch_gradients,
    inverse_soft_input_rejected=True,ema_teacher_has_no_gradient=True,demonstration_metrics={**metrics,**extra},branch_metrics=bm)),flush=True)
