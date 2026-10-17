"""Frozen training-only context intervention. No navigation or model update."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from feedback_objective import FeedbackData
from data import ROOT,sha

@torch.inference_mode()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    torch.set_num_threads(2);torch.manual_seed(651018)
    out=ROOT/'run_linear_evidence_v65/definition_context_audit.json'
    assert not out.exists()
    source=ROOT/'goal_discrimination_v64/source.pt'
    saved=torch.load(source,map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    data=FeedbackData(source.parent,saved['manifest']['transition_holdout_episodes'],torch.device('cuda'));del saved
    contexts=[]
    for start in range(0,len(data.features),32):
        x=data.features[start:start+32];o=hard(model.observe(x))
        contexts.append(model.context_sequence(x,o,data.previous[start:start+32],data.tasks[start:start+32]))
    contexts=torch.cat(contexts);fit_boundaries=sorted({(r[0],r[2]) for r in data.parts['fit']});results={}
    for task in (0,1):
        candidates=[(i,b) for i,b in fit_boundaries if int(data.tasks[i])==task]
        chosen=np.random.default_rng(651018+task).choice(len(candidates),min(128,len(candidates)),replace=False)
        selected=[candidates[i] for i in chosen]
        c=torch.stack([contexts[i,b] for i,b in selected]);definitions=F.normalize(model.observation_definitions[None]+model.observation_definition_delta(c),dim=-1)
        goal=model.goal_logits(c,torch.full((len(c),),task,device='cuda',dtype=torch.long))[1].argmax(-1)
        rows=[r for r in data.parts['holdout'] if int(data.tasks[r[0]])==task]
        i=torch.tensor([r[0] for r in rows],device='cuda');t=torch.tensor([r[1] for r in rows],device='cuda');b=torch.tensor([r[2] for r in rows],device='cuda')
        x=data.features[i,t];own_c=contexts[i,b];y=torch.tensor([r[3] for r in rows],device='cuda')
        own_goal=model.goal_logits(own_c,data.tasks[i])[1].argmax(-1)
        own_logits,z,_=model.observation_encoding(x,own_c);own=own_logits.argmax(-1)==own_goal
        logits=12*torch.einsum('iw,jow->ijo',z,definitions)
        equal=logits.argmax(-1)==goal[None];p=equal.float().mean(1)
        fractions={}
        for label,mask in [('positive',y),('negative',~y)]:
            values=p[mask]
            fractions[label]=dict(frames=int(mask.sum()),own_match=float(own[mask].float().mean()),
                across_context_mean_match=float(values.mean()),never_match=int((values==0).sum()),always_match=int((values==1).sum()),
                mixed_decisions=int(((values>0)&(values<1)).sum()),
                quantiles=torch.quantile(values,torch.tensor([0.,.25,.5,.75,1.],device='cuda')).cpu().tolist())
        results[['chair','table'][task]]=dict(training_contexts=len(c),heldout=fractions,
            generated_goal_ids=goal.unique(return_counts=True)[0].cpu().tolist())
    result=dict(time=time.time(),source_sha256=sha(source),results=results,optimizer_updates=0,new_rollouts=0,
        definitions_from_training_episodes=True,images_from_feedback_holdout_only=True,dev80_used=False,finalfixed80_used=False,
        caveat='Different training contexts are paired with the same current image and category solely to diagnose goal-equivalence invariance; not a policy execution.')
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
