"""Bounded CPU-only loss intervention; no deployment and no development test."""
import json,random,time
from pathlib import Path
import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.visual_graph_subgoal_training import objective
from predictive_nesy.visual_memory_subgoal_training import transition_objective
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
from predictive_nesy.data import sha

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'pilot_world_objective_v11';out.mkdir(exist_ok=True)
assert not (out/'candidate.pt').exists(),'Inspect existing pilot instead of repeating'
torch.set_num_threads(2);torch.manual_seed(1986);rng=random.Random(1986)
source=root/'run_vocabulary_rules_v11/checkpoint_002000.pt'
saved=torch.load(source,map_location='cpu',weights_only=False)
assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).train();m.load_state_dict(saved['model'])
for name,p in m.named_parameters():p.requires_grad_(name.startswith(('transitions.','subgoal.')))
fixed={n:v.clone() for n,v in m.state_dict().items() if not n.startswith(('transitions.','subgoal.'))}
windows,provenance=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
starts=[w['start'] for w in windows];groups={s:[w for w in windows if w['start']==s] for s in set(starts)}
banks=[TransitionBank(root/name,saved['manifest']['data']['train_layouts']) for name in [
    'counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5','counterfactual_proof_v8r1','counterfactual_proof_v9']]
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
began=time.time()
for step in range(1,301):
    start=rng.choice(starts);d=batch(rng.choices(groups[start],k=16),'cpu');opt.zero_grad()
    loss,metrics=objective(m,d,proof_from_predicted_state=True,proof_weight=0.)
    loss.backward();del d,loss
    bank=rng.choices(banks,weights=[len(b.keys) for b in banks],k=1)[0]
    loss,extra=transition_objective(m,bank.sample(rng,8,'cpu'));loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step()
    if step==1 or step%50==0:
        row=dict(update=step,seconds=time.time()-began,grad_norm=float(norm),**metrics,**extra)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
assert all(torch.equal(v,m.state_dict()[n]) for n,v in fixed.items())
payload=dict(saved,model=m.state_dict(),optimizer=opt.state_dict(),update=2300,
    pilot=dict(updates=300,source=str(source),proof_action_loss_weight=0.,cpu_only=True,not_deployed=True))
path=out/'candidate.pt';torch.save(payload,path)
path.with_suffix('.COMMITTED.json').write_text(json.dumps(dict(path=path.name,sha256=sha(path),update=2300,time=time.time()),indent=2))
(out/'complete.json').write_text(json.dumps(dict(updates=300,seconds=time.time()-began,frozen_weights_unchanged=True,not_deployed=True),indent=2))
print(json.dumps(dict(complete=True,seconds=time.time()-began)),flush=True)
