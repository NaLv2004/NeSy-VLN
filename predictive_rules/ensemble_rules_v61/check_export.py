import json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,SingleWorld,Config,definition_inputs,hard
from planner import Program,plan
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'diagnostic_ensemble_rules_v61'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(2);start=time.time()
saved=torch.load(ROOT/'onpolicy_successors_v60/source.pt',map_location='cpu',weights_only=False)
heads=torch.load(OUT/'heads.pt',map_location='cpu',weights_only=False)
m=FactoredWorld(Config(**saved['model_config'])).eval().requires_grad_(False);m.load_candidate(saved['model'],heads['heads'])
for k,v in saved['model'].items():assert torch.equal(m.state_dict()[k],v),k
source_frame=next((ROOT/'onpolicy_successors_v60/counterfactual_onpolicy').glob('*.npz'))
meta=json.loads(source_frame.with_suffix('.json').read_text());assert meta['training_only']
with np.load(source_frame,allow_pickle=False) as a:frame=torch.tensor(a['features'][:1].copy())[None]
task=torch.tensor([['chair','table'].index(meta['category'])])
with torch.no_grad():context=m.context_sequence(frame,hard(m.observe(frame)),torch.tensor([[17]]),task)[:,0]
with torch.no_grad():
    table=m.compile(context,task);program=Program(**table).validate()
    base=SingleWorld(m.cfg);base.load_state_dict(saved['model']);original=base.compile(context,task)
    assert np.array_equal(table['next_memory'],original['next_memory']) and table['goal']==original['goal']
    indices=torch.tensor([0,127,703,2047]);mi=F.one_hot(indices//64,32).float()[None]
    oi=F.one_hot(indices%64,64).float()[None];actions=torch.tensor([[1,2,5,16]])
    inputs=definition_inputs(m,context,mi,oi,actions)
    errors=[]
    for k,head in enumerate(m.member_rules):
        expected=head(*inputs).exp()[0].numpy();observed=table['outcome_probability'][k,indices.numpy(),actions[0].numpy()]
        errors.append(float(np.max(abs(expected-observed))));assert np.allclose(expected,observed,atol=2e-6)
    hooks=[module.register_forward_pre_hook(lambda *_:(_ for _ in ()).throw(RuntimeError('Neural planner call'))) for module in m.modules()]
    try:proof=plan(program,0,0,4)
    finally:
        for h in hooks:h.remove()
result=dict(time=time.time(),seconds=time.time()-start,unchanged_perception_goals_and_memory=True,
    all_member_row_errors=errors,planner_neural_calls=0,source=heads['source'],
    generated_goal_unchanged=True,max_actions=32,finalfixed80_used=False)
(OUT/'export_contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
