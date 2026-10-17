import json,random,time
from pathlib import Path
import torch
from predictive_nesy.graph_subgoal_model import GraphSubgoalModel,Config
from predictive_nesy.graph_subgoal_training import objective,transition_objective,proof_log_support,state_support
from predictive_nesy.graph_subgoal_model import straight_through
from predictive_nesy.graph_subgoal_interpreter import export_program,plan
import numpy as np
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank,canonical_provenance
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
saved=torch.load(root/'run_anonymous_v5/best_development.pt',map_location='cpu',weights_only=False);manifest=saved['manifest']
windows,provenance=load_windows(manifest['data']['source'],manifest['data']['train_layouts']);assert provenance==canonical_provenance(manifest['data'])
model=GraphSubgoalModel(Config()).cuda();current=model.state_dict()
for n,v in saved['model'].items():
    if n in current and not n.startswith('subgoal.') and v.shape==current[n].shape:current[n]=v
model.load_state_dict(current);rng=random.Random(18);data=batch(rng.choices([w for w in windows if w['start']==0],k=384),'cuda');begin=time.time()
with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,data)
loss.backward();assert torch.isfinite(loss);del loss,data
source=TransitionBank(root/'counterfactual_anonymous_v5',manifest['data']['train_layouts']);data=source.sample(rng,32,'cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=transition_objective(model,data)
loss.backward();assert torch.isfinite(loss);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
torch.cuda.synchronize();print(json.dumps(dict(passed=True,seconds=time.time()-begin,memory_gib=torch.cuda.max_memory_allocated()/2**30,metrics=metrics,counterfactual=extra)),flush=True)
data=batch(rng.choices([w for w in windows if w['start']==0],k=32),'cuda')
torch.backends.cuda.matmul.allow_tf32=True
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    p=model.generate(data['task'],data['history'][:,:1],data['previous_actions'][:,:1],data['history_length'],start=0)
    state=straight_through(model.ground(p,data['targets'][:,0]).squeeze(1));goal=straight_through(p['subgoal_logits'].squeeze(1).softmax(-1))
    support=state_support(state,proof_log_support(p,goal,10)[10]);choices=support.argmax(-1).cpu().tolist()
    compared=0
    for i in range(32):
        route=plan(export_program(p,i),np.array([state[i].argmax().item()]))
        if route['found']:
            assert choices[i]==route['actions'][0],(i,choices[i],route['actions'][0]);compared+=1
print(json.dumps(dict(cuda_bfloat16_tf32_exact_agreement=True,compared=compared)),flush=True)
