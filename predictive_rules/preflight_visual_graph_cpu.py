"""Bounded actual-data gradient check without occupying the evaluation GPU."""
import json,random,time
from pathlib import Path
import torch
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel,Config
from predictive_nesy.visual_graph_subgoal_training import objective,transition_objective
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
saved=torch.load(root/'run_graph_subgoal_v3/checkpoint_000400.pt',map_location='cpu',weights_only=False)
model=VisualGraphSubgoalModel(Config());current=model.state_dict()
for name,value in saved['model'].items():
    preserve=name=='offsets' or name.startswith(('tokens.','observation.','context_read.','history.','task.','action_embed.','context_norm.'))
    if preserve and name in current and value.shape==current[name].shape:current[name]=value
model.load_state_dict(current);manifest=saved['manifest'];rng=random.Random(18)
windows,_=load_windows(manifest['data']['source'],manifest['data']['train_layouts'])
data=batch(rng.choices([w for w in windows if w['start']==0],k=16),'cpu');begin=time.time()
loss,metrics=objective(model,data);loss.backward();assert torch.isfinite(loss)
source=TransitionBank(root/'counterfactual_anonymous_v5',manifest['data']['train_layouts'])
loss,extra=transition_objective(model,source.sample(rng,4,'cpu'));loss.backward();assert torch.isfinite(loss)
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
print(json.dumps(dict(passed=True,seconds=time.time()-begin,metrics=metrics,transitions=extra,gradient_norm=float(norm))))
