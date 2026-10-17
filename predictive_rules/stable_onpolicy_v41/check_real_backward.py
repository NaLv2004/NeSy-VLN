import json
from pathlib import Path
import torch
from model import FactoredWorld,Config
from objectives import objective
torch.set_num_threads(3)
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
case=torch.load(root/'run_induced_effects_v39/backward_case_rank0.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**case['model_config']))
model.load_state_dict(case['model'])
data={key:value if key=='operator_targets' else value[:2] for key,value in case['data'].items()}
loss,metrics=objective(model,data);loss.backward()
unused=[name for name,p in model.named_parameters() if p.grad is None]
assert not unused,unused
assert all(torch.isfinite(p.grad).all() for p in model.parameters())
norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
print(json.dumps(dict(real_training_batch=True,loss=float(loss),gradient_norm=norm,
    finite_gradients=True,all_training_parameters_used=True)))
