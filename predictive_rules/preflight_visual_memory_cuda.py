import json,random,time
from pathlib import Path
import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.visual_memory_subgoal_training import objective,transition_objective
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
from test_visual_memory_subgoal import run

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.9)
run('cuda');torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
saved=torch.load(root/'run_visual_graph_subgoal_v5/checkpoint_004000.pt',map_location='cpu',weights_only=False)
model=VisualMemorySubgoalModel(Config(**saved['model_config'])).cuda()
result=model.load_state_dict(saved['model'],strict=False)
assert not result.unexpected_keys and all(k.startswith(('visual_memory.','memory_projection.')) for k in result.missing_keys)
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
rng=random.Random(72);torch.backends.cuda.matmul.allow_tf32=True
bank=TransitionBank(root/'counterfactual_anonymous_v5',saved['manifest']['data']['train_layouts'])
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
begin=time.time();data=batch(rng.choices([w for w in windows if w['start']==10],k=640),'cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,data)
loss.backward();del loss,data
data=bank.sample(rng,32,'cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=transition_objective(model,data)
loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
assert torch.isfinite(loss) and torch.isfinite(norm)
optimizer.step();torch.cuda.synchronize()
result=dict(passed=True,real_batch_size=640,transition_batch=32,seconds=time.time()-begin,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,grad_norm=float(norm),metrics=metrics,transitions=extra,
    no_checkpoint_written=True,training_split=saved['manifest']['data']['train_layouts'])
(root/'diagnostics/visual_memory_v6_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
