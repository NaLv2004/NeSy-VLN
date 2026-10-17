"""One actual-data training pass before starting the independent distributed job."""
import json,random,time
from pathlib import Path
import torch
from predictive_nesy.environment import make_bank
from predictive_nesy.subgoal_model import SubgoalRuleModel,Config
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank,canonical_provenance
from predictive_nesy.subgoal_training import objective,transition_objective
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
saved=torch.load(root/'run_anonymous_v5/best_development.pt',map_location='cpu',weights_only=False);bank=make_bank()
windows,provenance=load_windows(saved['manifest']['data']['source'],bank.provenance['train_layouts'])
assert provenance==canonical_provenance(saved['manifest']['data'])
model=SubgoalRuleModel(Config(**saved['model_config'])).cuda();state=model.state_dict()
for name,value in saved['model'].items():
    if name in state and not name.startswith('subgoal.') and value.shape==state[name].shape:state[name]=value
model.load_state_dict(state);rng=random.Random(44);data=batch([w for w in windows if w['start']==0][:2],'cuda');begin=time.time()
with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,data)
loss.backward();print(json.dumps(dict(demonstrations=metrics,windows=len(windows))),flush=True)
for path in ['counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5']:
    model.zero_grad(set_to_none=True);bank2=TransitionBank(root/path,bank.provenance['train_layouts']);data=bank2.sample(rng,2,'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=transition_objective(model,data)
    loss.backward();assert torch.isfinite(loss);print(json.dumps(dict(bank=path,states=len(bank2.keys),metrics=metrics)),flush=True)
print(json.dumps(dict(passed=True,seconds=time.time()-begin,memory_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
