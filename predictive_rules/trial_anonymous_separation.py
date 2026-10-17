"""One bounded training-only representation trial; no privileged state labels."""
import copy,hashlib,json,os,random,time
from pathlib import Path
import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.visual_graph_subgoal_training import objective
from predictive_nesy.visual_memory_subgoal_training import transition_objective
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'anonymous_separation_trial'
if (out/'launcher.json').exists():raise RuntimeError('Trial already launched; do not duplicate')
out.mkdir(exist_ok=True);torch.set_num_threads(2)
if os.environ.get('CUDA_VISIBLE_DEVICES')!='7':raise RuntimeError('This bounded trial is allocated GPU7 only')
torch.manual_seed(923);rng=random.Random(923)
source=root/'run_visual_memory_subgoal_v6/checkpoint_000400.pt'
saved=torch.load(source,map_location='cpu',weights_only=False)
model=VisualMemorySubgoalModel(Config(**saved['model_config'])).cuda();model.load_state_dict(saved['model'])
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
starts=[w['start'] for w in windows];groups={s:[w for w in windows if w['start']==s] for s in set(starts)}
banks=[TransitionBank(root/name,saved['manifest']['data']['train_layouts']) for name in
       ['counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5']]
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
manifest=copy.deepcopy(saved['manifest']);manifest.update(architecture='anonymous_rollout_proof_subgoal_v8',
    state_separation_intervention=dict(source=str(source),updates=600,learning_rate=1e-4,weight=10.,
        targets='pairs of actually different visual token arrays within each observed sequence',
        manually_defined_predicate_meanings=False,training_only=True),
    proof_state_source='initial state and rule propagation with demonstrated action prefix',
    recurrent_precision='fp32',reward_table=False,value_table=False,direct_action_head=False)
(out/'launcher.json').write_text(json.dumps(dict(pid=os.getpid(),gpu=7,time=time.time(),source=str(Path(__file__).parent))))
begin=time.time()
for step in range(1,601):
    optimizer.zero_grad(set_to_none=True);model.train();start=rng.choice(starts)
    data=batch(rng.choices(groups[start],k=512),'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,data,proof_from_predicted_state=True,state_separation_weight=10.)
    loss.backward();del loss,data
    data=rng.choice(banks).sample(rng,32,'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=transition_objective(model,data,state_separation_weight=10.)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite separation-trial gradient')
    optimizer.step();del loss,data
    if step%100==0:
        torch.cuda.synchronize();metrics.update(extra,step=step,seconds=time.time()-begin,gradient_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_config=model.export_config(),manifest=manifest,update=600),out/'candidate.tmp')
os.replace(out/'candidate.tmp',out/'candidate.pt')
commit=dict(path='candidate.pt',sha256=hashlib.sha256((out/'candidate.pt').read_bytes()).hexdigest(),update=600,time=time.time())
(out/'candidate.COMMITTED.json').write_text(json.dumps(commit,indent=2))
print(json.dumps(dict(completed=True,seconds=time.time()-begin,checkpoint=commit)),flush=True)
