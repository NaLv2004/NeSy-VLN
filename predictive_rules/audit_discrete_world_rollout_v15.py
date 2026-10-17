"""Offline drift audit; later observations are labels, never rollout inputs.

Uses an immutable component checkpoint and training-corpus episode holdout.
This is not navigation validation and does not change a live controller.
"""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from predictive_nesy.discrete_visual_world import DiscreteVisualWorld,WorldConfig
from predictive_nesy.subgoal_data import load_windows
from predictive_nesy.data import sha

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in usage.splitlines()}[7]<300
out=ROOT/'diagnostics/discrete_world_600_rollout_v15'
assert not out.exists();out.mkdir();torch.set_num_threads(2)
path=ROOT/'run_discrete_visual_world_v15/checkpoint_000600.pt'
commit=json.loads(path.with_suffix('.COMMITTED.json').read_text());assert sha(path)==commit['sha256']
saved=torch.load(path,map_location='cpu',weights_only=False)
model=DiscreteVisualWorld(WorldConfig(**saved['model_config'])).cuda().eval()
model.load_state_dict(saved['model']);model.requires_grad_(False)
dm=saved['manifest']['data'];windows,_=load_windows(dm['data']['source'],dm['data']['train_layouts'])
held=[w for w in windows if int(hashlib.sha256(w['episode_id'].encode()).hexdigest()[:8],16)%5==0]
accumulators={t:dict(patches=0,autonomous_correct=0,one_step_correct=0,copy_correct=0,
                     changed_patches=0,autonomous_changed_correct=0,one_step_changed_correct=0,
                     windows=0,autonomous_exact=0) for t in range(1,11)}
start=time.time()
with torch.inference_mode():
    for begin in range(0,len(held),128):
        selected=held[begin:begin+128]
        targets=torch.tensor(np.asarray([w['targets'] for w in selected]),device='cuda')
        actions=torch.tensor(np.asarray([w['actions'] for w in selected]),device='cuda')
        valid=torch.tensor(np.asarray([w['action_valid'] for w in selected]),device='cuda')
        state=targets[:,0].clone()
        for t in range(10):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                # This branch never reads an executed future observation.
                state=model(state,actions[:,t]).argmax(-1)
                # Separate offline diagnostic only: true predecessor is a label.
                one_step=model(targets[:,t],actions[:,t]).argmax(-1)
            mask=valid[:,t];truth=targets[mask,t+1];changed=truth!=targets[mask,t]
            match=state[mask]==truth;one=one_step[mask]==truth;r=accumulators[t+1]
            for key,value in dict(patches=truth.numel(),autonomous_correct=int(match.sum()),
                    one_step_correct=int(one.sum()),copy_correct=int((targets[mask,0]==truth).sum()),
                    changed_patches=int(changed.sum()),autonomous_changed_correct=int((match&changed).sum()),
                    one_step_changed_correct=int((one&changed).sum()),windows=int(mask.sum()),
                    autonomous_exact=int(match.all(-1).sum())).items():r[key]+=value
    # Action discrimination: which measured action outcome has the highest
    # probability under that action's own predicted categorical distribution?
    with np.load(ROOT/'discrete_visual_world_data_v15/transitions.npz',allow_pickle=False) as data:
        indices=np.flatnonzero(data['held'] & (data['kind']==0))
        # Banks store 17 adjacent measured branches per actual predecessor.
        assert len(indices)%17==0
        before=data['before'][indices].reshape(-1,17,49)
        after=data['after'][indices].reshape(-1,17,49)
        assert (before==before[:,:1]).all()
        assert (data['actions'][indices].reshape(-1,17)==np.arange(17)).all()
    confusion=np.zeros((17,17),np.int64)
    for begin in range(0,len(before),64):
        source=torch.tensor(before[begin:begin+64,0],device='cuda').long()
        target=torch.tensor(after[begin:begin+64],device='cuda').long()
        b=len(source);a=torch.arange(17,device='cuda').repeat(b)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(source.repeat_interleave(17,0),a).float().log_softmax(-1).reshape(b,17,49,64)
        score=torch.zeros((b,17,17),device='cuda')
        for actual in range(17):
            score[:,actual]=logits.gather(-1,target[:,actual,None,:,None].expand(-1,17,-1,-1)).squeeze(-1).mean(-1)
        prediction=score.argmax(-1).cpu().numpy()
        for actual in range(17):np.add.at(confusion[actual],prediction[:,actual],1)
rows=[]
for t,r in accumulators.items():
    rows.append(dict(step=t,**r,autonomous_accuracy=r['autonomous_correct']/max(r['patches'],1),
        one_step_accuracy=r['one_step_correct']/max(r['patches'],1),
        unchanged_start_baseline=r['copy_correct']/max(r['patches'],1),
        autonomous_changed_accuracy=r['autonomous_changed_correct']/max(r['changed_patches'],1),
        one_step_changed_accuracy=r['one_step_changed_correct']/max(r['changed_patches'],1)))
result=dict(source=str(path),source_sha256=sha(path),training_only=True,no_navigation_validation=True,
    internal_holdout_windows=len(held),future_observation_used_for_autonomous_rollout=False,
    perception_pretrained_on_source_corpus=True,per_horizon=rows,
    action_discrimination_confusion=confusion.tolist(),action_discrimination=float(np.trace(confusion)/confusion.sum()),
    movement_action_discrimination=float(np.diag(confusion)[1:].sum()/confusion[1:].sum()),
    identical_measured_outcomes_may_make_actions_ambiguous=True,seconds=time.time()-start,
    navigation_success_rate=None,no_model_deployed=True)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
