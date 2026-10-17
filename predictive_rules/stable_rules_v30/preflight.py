"""Verify selected-row equivalence, stable state labels and hard execution."""
import copy
import importlib.util
import json
import random
import torch
from planning_model import PlanningObservedWorld, Config
from stable_objectives import observed_objective, branch_representation_loss
from train import ROOT, Windows, prepare
from predictive_nesy.subgoal_data import batch, TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures

torch.set_num_threads(2);torch.manual_seed(30);rng=random.Random(3021)
saved=torch.load(ROOT/'run_planning_observed_v29/checkpoint_002000.pt',map_location='cpu',weights_only=False)
model=PlanningObservedWorld(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
prefixes=('definition_rules.','subgoal.','raw_','goal_','rule_evidence.','revision.','correction.')
for name,p in model.named_parameters():p.requires_grad_(name.startswith(prefixes))
frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
corpus=Windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cpu')
rows=[corpus.window(*key) for key in rng.choices(corpus.by_start[4],k=2)];d=batch(rows,'cpu');p=prepare(model,d,features,rows)
spec=importlib.util.spec_from_file_location('v29_reference',ROOT/'src_planning_observed_v29/train.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
old_loss,old_metrics,old_ids=reference.observed_objective(model,d,p)
new_loss,new_metrics,new_ids=observed_objective(model,d,p)
assert torch.equal(old_ids,new_ids)
assert torch.allclose(old_loss,new_loss,atol=1e-5,rtol=1e-5),(old_loss,new_loss)
for key in old_metrics:
    assert abs(old_metrics[key]-new_metrics[key])<1e-5,(key,old_metrics[key],new_metrics[key])
del old_loss
before_ids=model.encode_sequence(p,d['targets'])[0].argmax(-1).detach().clone()
before_definitions=p['definitions'].detach().clone()
model.zero_grad(set_to_none=True);new_loss.backward()
gradient={k:sum(float(v.grad.norm()) for n,v in model.named_parameters() if n.startswith(k) and v.grad is not None)
          for k in ['definition_rules.','rule_evidence.','subgoal.','observation.','definitions.','visual_memory.']}
assert gradient['definition_rules.']>0 and gradient['subgoal.']>0
assert gradient['observation.']==gradient['definitions.']==gradient['visual_memory.']==0
opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4)
torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.);opt.step()
after=prepare(model,d,features,rows)
assert torch.equal(before_definitions,after['definitions'])
assert torch.equal(before_ids,model.encode_sequence(after,d['targets'])[0].argmax(-1))
assert all(torch.equal(v,dict(model.named_parameters())[n]) for n,v in frozen.items())
bank=TransitionBank(saved['manifest']['transition_banks'][0]['path'],saved['manifest']['data']['train_layouts'])
branches=bank.sample(rng,2,'cpu');raw,available=features.transitions(branches,bank.directory)
bp=model.prepare(branches['task'],branches['history'],branches['previous_actions'],torch.full((2,),branches['history'].shape[1],dtype=torch.long),start=branches['start'],boundary_features=raw,boundary_available=available)
branch_loss,branch_metrics=branch_representation_loss(model,None,bp,None,branches)
assert torch.isfinite(branch_loss);model.zero_grad(set_to_none=True);branch_loss.backward()
assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith('definition_rules.'))
from check_interpreter import main
main()
result=dict(passed=True,selected_row_objective_matches_v29=True,hard_state_labels_unchanged_after_optimizer_update=True,
            all_frozen_parameters_unchanged=True,gradient_attribution=gradient,branch_metrics=branch_metrics,
            training_only=True,source_checkpoint_unchanged=True,diagnostic_optimizer_updates=1)
(ROOT/'stable_observed_rules_v30_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
