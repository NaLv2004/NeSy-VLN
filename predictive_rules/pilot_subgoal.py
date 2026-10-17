"""Bounded training-only trial of rule-based inverse-action discrimination."""
import json,random,time
from pathlib import Path
import torch
from predictive_nesy.subgoal_model import SubgoalRuleModel,Config
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
from predictive_nesy.subgoal_training import objective,transition_objective
from train_subgoal import save,atomic
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'subgoal_rule_inverse_trial';out.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(1921);rng=random.Random(1921)
saved=torch.load(root/'run_subgoal_v1/checkpoint_001200.pt',map_location='cuda',weights_only=False)
model=SubgoalRuleModel(Config(**saved['model_config'])).cuda();model.load_state_dict(saved['model'])
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);optimizer.load_state_dict(saved['optimizer'])
manifest=saved['manifest'];manifest=dict(manifest,training_revision=2,rule_inverse_supervision=True,pilot_source_update=1200)
windows,_=load_windows(manifest['data']['source'],manifest['data']['train_layouts']);starts=[w['start'] for w in windows]
by_start={s:[w for w in windows if w['start']==s] for s in set(starts)}
sources=[TransitionBank(root/name,manifest['data']['train_layouts']) for name in ['counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5']]
began=time.time()
for update in range(1,401):
    start=rng.choice(starts);optimizer.zero_grad(set_to_none=True);data=batch(rng.choices(by_start[start],k=64),'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,data)
    loss.backward();del loss,data
    source=rng.choice(sources);data=source.sample(rng,16,'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=transition_objective(model,data)
    loss.backward();metrics.update(extra);del loss,data
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
    if update%100==0:
        metrics.update(pilot_update=update,seconds=time.time()-began);print(json.dumps(metrics),flush=True)
        atomic(out/'status.json',metrics)
state=dict(python=rng.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
save(out/'candidate.pt',model,optimizer,1600,manifest,[state]);print('CANDIDATE_COMMITTED',flush=True)
