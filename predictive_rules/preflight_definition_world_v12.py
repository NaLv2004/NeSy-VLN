import json,random,time,torch
from pathlib import Path
from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld,Config
from predictive_nesy.rollout_proof_training import objective,transition_objective
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
assert json.loads((root/'initialization_definition_world_v12/report.json').read_text())['complete']
saved=torch.load(root/'initialization_definition_world_v12/initial.pt',map_location='cpu',weights_only=False)
m=DefinitionConditionedWorld(Config(**saved['model_config'])).cuda();m.load_state_dict(saved['model'])
for name,p in m.named_parameters():p.requires_grad_(name.startswith(('definition_rules.','subgoal.')))
fixed={name:value.detach().cpu().clone() for name,value in m.state_dict().items() if not name.startswith(('definition_rules.','subgoal.'))}
before=m.definition_rules.query[-1].weight.detach().clone();rng=random.Random(491)
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
bank=TransitionBank(root/'counterfactual_proof_v11',saved['manifest']['data']['train_layouts'])
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4);begin=time.time()
d=batch(rng.choices([w for w in windows if w['start']==0],k=640),'cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(m,d)
loss.backward();del loss,d
d=bank.sample(rng,192,'cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=transition_objective(m,d)
loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm)
opt.step();torch.cuda.synchronize()
assert all(torch.equal(value,m.state_dict()[name].detach().cpu()) for name,value in fixed.items())
assert not torch.equal(before,m.definition_rules.query[-1].weight)
result=dict(passed=True,frozen_representation_bitwise_identical=True,rules_updated=True,gradient_norm=float(norm),
    seconds=time.time()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
    demonstration_windows=640,counterfactual_states=192,metrics=metrics,counterfactual=extra)
(root/'diagnostics/definition_world_v12_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
