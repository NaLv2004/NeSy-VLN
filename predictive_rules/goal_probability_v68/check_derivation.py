"""Compare differentiable fixed-sequence derivation against explicit tables."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config
from joint_objective import derived_probabilities
from data import ROOT,sha

def main():
    torch.set_num_threads(2);torch.manual_seed(681017)
    src=ROOT/'goal_discrimination_v64';saved=torch.load(src/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model']);del saved
    # Real program generation contexts from completed training trajectories.
    path=next((ROOT/'counterfactual_closure_v63/episodes').glob('*.npz'))
    with np.load(path) as data:context=torch.tensor(data['context'][:3]).float()
    results=[]
    for i,ctx in enumerate(context):
        ctx=ctx[None];task=torch.tensor([i%2]);program=model.compile(ctx,task);gm,go=divmod(program['goal'],64)
        for a1,a2 in [(0,1),(1,0),(3,7),(10,16)]:
            mi=i+3;oi=i+9
            m=F.one_hot(torch.tensor([mi]),32).float();o=F.one_hot(torch.tensor([oi]),64).float()
            p1,p2=derived_probabilities(model,ctx,m,o,torch.tensor([a1]),torch.tensor([a2]),torch.tensor([gm]),torch.tensor([go]))
            tm=program['next_memory'];tp=program['outcome_probability'];s=mi*64+oi;m1=tm[s,a1]
            first=float(tp[s,a1,go])*(float(m1==gm) if a1==0 else float(tm[m1*64+go,0]==gm))
            second=0.
            for ob in range(64):
                s1=m1*64+ob;m2=tm[s1,a2]
                terminal=float(m2==gm) if a2==0 else float(tm[m2*64+go,0]==gm)
                second+=float(tp[s,a1,ob])*float(tp[s1,a2,go])*terminal
            assert np.isclose(float(p1),first,atol=2e-6),(p1,first)
            assert np.isclose(float(p2),second,atol=2e-6),(p2,second)
            results.append(dict(first_action=a1,second_action=a2,one_error=abs(float(p1)-first),two_error=abs(float(p2)-second)))
    model.zero_grad();(p1+p2).sum().backward()
    gradients={k:float(v.grad.norm()) for k,v in model.named_parameters() if v.grad is not None and float(v.grad.norm())>0}
    assert gradients and any(k.startswith('definition_rule.') for k in gradients)
    result=dict(time=time.time(),cases=len(results),max_absolute_error=max(max(r['one_error'],r['two_error']) for r in results),
        nonzero_gradient_parameters=len(gradients),state_forward='hard categorical IDs',neural_value_or_action_head=False,
        formula='Sum of existing categorical transition probabilities times exact generated joint-goal membership',
        first_stop_cases='Second probability mathematically checked but masked out of training for terminal first actions',training_updates=0,new_rollouts=0)
    out=ROOT/'goal_probability_v68';out.mkdir(exist_ok=True);(out/'derivation_contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
