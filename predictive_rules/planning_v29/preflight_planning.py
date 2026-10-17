"""Real-data gradient attribution and exact categorical training boundaries."""
import copy,json,random
import torch
from planning_model import PlanningObservedWorld,Config,representation_loss,perception_proof_loss,branch_representation_loss
from train import ROOT,Windows,prepare,observed_objective
from predictive_nesy.subgoal_data import batch,TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures

torch.set_num_threads(2);torch.manual_seed(29);rng=random.Random(2916)
saved=torch.load(ROOT/'run_adaptive_observed_v27/checkpoint_001000.pt',map_location='cpu',weights_only=False)
model=PlanningObservedWorld(Config(**saved['model_config']));missing,unexpected=model.load_state_dict(saved['model'],strict=False)
assert not unexpected and all(k.startswith('inverse_discrete.') for k in missing)
teacher=copy.deepcopy(model).eval().requires_grad_(False)
corpus=Windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cpu')
rows=[corpus.window(*k) for k in rng.choices(corpus.by_start[4],k=4)];data=batch(rows,'cpu');p=prepare(model,data,features,rows)
loss,pm=perception_proof_loss(model,p,data);loss.backward()
def gradients():
    return {k:sum(float(v.grad.norm()) for n,v in model.named_parameters() if n.startswith(k) and v.grad is not None)
            for k in ['observation.','tokens.','definitions.','visual_memory.','memory_projection.','definition_rules.','subgoal.','inverse_discrete.']}
proof_gradients=gradients()
assert all(proof_gradients[k]>0 for k in ['observation.','tokens.','definitions.','visual_memory.','memory_projection.']),proof_gradients
assert proof_gradients['definition_rules.']==proof_gradients['subgoal.']==proof_gradients['inverse_discrete.']==0
model.zero_grad(set_to_none=True)
bank=TransitionBank(saved['manifest']['transition_banks'][0]['path'],saved['manifest']['data']['train_layouts'])
d=bank.sample(rng,4,'cpu');raw,available=features.transitions(d,bank.directory);lengths=torch.full((4,),d['history'].shape[1],dtype=torch.long)
kw=dict(start=d['start'],boundary_features=raw,boundary_available=available)
p=model.prepare(d['task'],d['history'],d['previous_actions'],lengths,**kw)
with torch.no_grad():tp=teacher.prepare(d['task'],d['history'],d['previous_actions'],lengths,**kw)
loss,bm=branch_representation_loss(model,teacher,p,tp,d);loss.backward();branch_gradients=gradients()
assert all(branch_gradients[k]>0 for k in ['observation.','tokens.','definitions.','visual_memory.','definition_rules.','inverse_discrete.']),branch_gradients
assert torch.isfinite(loss)
from check_interpreter import main
main()
print(json.dumps(dict(passed=True,proof_gradient_attribution=proof_gradients,branch_gradient_attribution=branch_gradients,
                     proof_metrics=pm,branch_metrics=bm,optimizer_updates=0)),flush=True)
