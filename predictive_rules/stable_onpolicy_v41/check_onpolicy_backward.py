"""Check the new loss on immutable real off-policy states before training."""
import json
from pathlib import Path
import numpy as np
import torch
from model import FactoredWorld,Config
from objectives import objective

torch.set_num_threads(3)
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
cache=root/'onpolicy_successors_v40'
saved=torch.load(cache/'policy.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config']))
model.load_state_dict(saved['model'],strict=True)
case=torch.load(root/'run_induced_effects_v39/backward_case_rank0.pt',map_location='cpu',weights_only=False)
data={key:value if key=='operator_targets' else value[:2] for key,value in case['data'].items()}
records=sorted((cache/'counterfactual_onpolicy').glob('*.json'))[:2]
assert len(records)==2
for b,record_path in enumerate(records):
    record=json.loads(record_path.read_text());assert record['training_only']
    with np.load(record_path.with_suffix('.npz')) as a:
        n=len(a['features']);t=n-1
        data['branch_features'][b].zero_();data['branch_previous'][b].fill_(17)
        data['branch_features'][b,:n]=torch.from_numpy(a['features'])
        data['branch_previous'][b,1:n]=torch.from_numpy(a['recorded_actions'][:n-1])
        data['branch_t'][b]=t;data['branch_start'][b]=max(0,t-6)
        data['branch_task'][b]=['chair','table'].index(record['category'])
        data['successors'][b]=torch.from_numpy(a['successor_features'][t])
data['branch_onpolicy']=torch.ones(2,dtype=torch.bool)
rows=[]
for phase in [False,True]:
    model.zero_grad(set_to_none=True);torch.manual_seed(41)
    data['world_updates_encoder']=phase
    loss,metrics=objective(model,data);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    rows.append(dict(phase=phase,loss=float(loss.detach()),gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5)),
        **{key:float(metrics[key]) for key in ['branch_loss','visual_vq_loss','memory_vq_loss','onpolicy_branch_observation_accuracy','onpolicy_branch_fraction']}))
assert rows[0]['loss']==rows[1]['loss']
result=dict(checkpoint_update=saved['update'],training_episodes=[p.stem for p in records],actual_onpolicy_states=True,
    all_parameters_finite=True,same_forward_loss=True,phases=rows)
Path(__file__).with_name('onpolicy_backward.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
