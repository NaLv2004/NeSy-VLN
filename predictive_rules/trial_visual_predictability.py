"""One bounded training-only intervention on learned-state predictability."""
import json,random,time,hashlib,os
from pathlib import Path
import torch
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel,Config
from predictive_nesy.visual_graph_predictive_training import objective,transition_objective
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'visual_predictability_trial'
if (out/'candidate.pt').exists():raise RuntimeError('Bounded trial already finished')
out.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
saved=torch.load(root/'run_visual_graph_subgoal_v5/checkpoint_002000.pt',map_location='cpu',weights_only=False)
model=VisualGraphSubgoalModel(Config(**saved['model_config'])).cuda();model.load_state_dict(saved['model'])
manifest=dict(saved['manifest']);windows,_=load_windows(manifest['data']['source'],manifest['data']['train_layouts'])
starts=[w['start'] for w in windows];by_start={s:[w for w in windows if w['start']==s] for s in set(starts)}
bank=TransitionBank(root/'counterfactual_anonymous_v5',manifest['data']['train_layouts'])
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);rng=random.Random(749);torch.manual_seed(749)
begin=time.time();manifest['predictability_intervention']=dict(source_update=2000,updates=400,posterior_alignment_weight=1.,
    planning_premise_gradient=False,train_only=True,no_runtime_observation_correction=True,learning_rate=1e-4)
(out/'launcher.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),gpu=7,source=str(Path(__file__).parent))))
for step in range(1,401):
    start=rng.choice(starts);optimizer.zero_grad(set_to_none=True);model.train()
    data=batch(rng.choices(by_start[start],k=128),'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,data)
    loss.backward();del loss,data
    data=bank.sample(rng,8,'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=transition_objective(model,data)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
    optimizer.step();del loss,data
    if step%50==0:
        torch.cuda.synchronize();metrics.update(step=step,seconds=time.time()-begin,gradient_norm=float(norm),memory_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_config=model.export_config(),manifest=manifest,
    update=2400),out/'candidate.tmp');os.replace(out/'candidate.tmp',out/'candidate.pt')
commit=dict(path='candidate.pt',sha256=hashlib.sha256((out/'candidate.pt').read_bytes()).hexdigest(),update=2400,time=time.time())
(out/'candidate.COMMITTED.json').write_text(json.dumps(commit,indent=2));print(json.dumps(dict(completed=True,seconds=time.time()-begin,checkpoint=commit)),flush=True)
