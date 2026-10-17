"""Training-only diagnosis of goal-code aliasing. No executed policy actions."""
import json,os,sys,time,select
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_counterfactual_closure_v63'))
from model import FactoredWorld,Config
from data import sha

def summarize(y,pred,prob):
    tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    return dict(positions=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        true_positive_rate=tp/max(1,tp+fn),false_positive_rate=fp/max(1,fp+tn),
        precision=tp/max(1,tp+fp),balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)),
        positive_goal_probability=float(prob[y].mean()) if y.any() else None,
        negative_goal_probability=float(prob[~y].mean()) if (~y).any() else None)

@torch.inference_mode()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7';torch.set_num_threads(2)
    out=ROOT/'goal_discrimination_v64';old=ROOT/'counterfactual_closure_v63'
    status=json.loads((out/'status.json').read_text())
    if status['state']=='collecting':
        pid=status['pid'];command=Path(f'/proc/{pid}/cmdline')
        assert command.exists() and b'src_goal_discrimination_v64/coordinate_collection.py' in command.read_bytes()
        fd=os.pidfd_open(pid)
        try:select.select([fd],[],[],60)
        finally:os.close(fd)
        status=json.loads((out/'status.json').read_text())
    assert status['state']=='complete',status
    assert not (out/'goal_alias_audit.json').exists(),'Do not repeat a completed audit'
    commit=json.loads((out/'source.COMMITTED.json').read_text());assert sha(out/'source.pt')==commit['sha256']
    saved=torch.load(out/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    holdout=set(saved['manifest']['transition_holdout_episodes']);del saved
    observations=[];details=[];began=time.time()
    for meta in sorted((out/'episodes').glob('*.json')):
        r=json.loads(meta.read_text());assert r['training_only']
        name=r['episode']
        with np.load(meta.with_suffix('.npz'),allow_pickle=False) as d:features=d['features'].copy();y=d['stop_success'].copy()
        with np.load(old/'episodes'/(name+'.npz'),allow_pickle=False) as d:context=d['context'].copy()
        # Only recorded decision positions, excluding the absorbing frame after
        # actual STOP and any extra frame for which no rule context was stored.
        n=len(context);assert len(features)>=n
        x=torch.from_numpy(features[:n]).cuda();c=torch.from_numpy(context).cuda()
        task=torch.full((n,),['chair','table'].index(r['category']),device='cuda',dtype=torch.long)
        lo=model.observe(x,c);gm,go=model.goal_logits(c,task)
        goal=go.argmax(-1);matches=(lo.argmax(-1)==goal).cpu().numpy()
        probability=lo.softmax(-1).gather(-1,goal[:,None]).squeeze(-1).cpu().numpy()
        observations.extend(dict(part='holdout' if name in holdout else 'fit',eligible=bool(y[t]),
            predicted=bool(matches[t]),probability=float(probability[t])) for t in range(n))
        details.append(dict(episode=name,part='holdout' if name in holdout else 'fit',domain=r['domain'],
            **summarize(y[:n],matches,probability)))
    result=dict(time=time.time(),seconds=time.time()-began,source=commit,
        claim='Goal observation-code discrimination only, not actual STOP-policy performance or navigation success',
        contexts='Existing causal program-boundary contexts, frozen in v63; no future context',
        no_new_navigation_trial=True,finalfixed80_used=False,dev80_used=False,rows=details,summary={})
    for part in ('fit','holdout'):
        rows=[r for r in observations if r['part']==part]
        result['summary'][part]=summarize(np.array([r['eligible'] for r in rows]),
            np.array([r['predicted'] for r in rows]),np.array([r['probability'] for r in rows]))
    (out/'goal_alias_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()
