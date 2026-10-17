"""Compare real sequential versus batched VLM/vision processing across a revision boundary."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstration,demonstrations_many
from nesy_vla.training import train_batch

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
cfg=read('configs/navigation.json');setup(cfg);bank=DataBank(cfg);torch.manual_seed(cfg['seed'])
model=NavigationModel(cfg).cuda().eval();state=load_checkpoint(a.checkpoint,model)
episode_ids=[next(e for e in bank.demo_ids if 14<=len(bank.demos[e])<=18),next(e for e in bank.demo_ids if len(bank.demos[e])==8)]
for e in episode_ids:demonstration(model,bank,e)
demonstrations_many(model,bank,episode_ids)
torch.cuda.synchronize()
start=time.time();serial=[demonstration(model,bank,e) for e in episode_ids];torch.cuda.synchronize();serial_s=time.time()-start
start=time.time();parallel=demonstrations_many(model,bank,episode_ids);torch.cuda.synchronize();parallel_s=time.time()-start
checks=[]
for e,x,y in zip(episode_ids,serial,parallel):
    first=[r for c in x.chunks for r in c['rows']];second=[r for c in y.chunks for r in c['rows']]
    assert [r['action'] for r in first]==[r['action'] for r in second]
    assert [c['start'] for c in x.chunks]==[c['start'] for c in y.chunks]
    errors={key:max(float(np.max(np.abs(np.asarray(u[key])-np.asarray(v[key])))) for u,v in zip(first,second))
            for key in ['belief','confidence','support','old_logp']}
    assert max(errors.values())<.05,errors
    checks.append({'episode_id':e,'steps':len(first),'revision_starts':[c['start'] for c in x.chunks],'maximum_absolute_errors':errors})
result={'checkpoint_update':state['update'],'sequential_seconds':serial_s,'batched_seconds':parallel_s,
        'cases':checks,'passed':True,'note':'BF16 kernels allow numerical differences; actions here are expert labels, not autonomous successes.'}
model.train();chunks=[(c,True) for p in parallel for c in p.chunks if c['rows']]
loss,_=train_batch(model,chunks);loss.backward()
result['backward_loss']=float(loss.detach())
result['vlm_gradient_norm']=sum(float(p.grad.float().square().sum()) for p in model.generator.vlm.parameters() if p.grad is not None)**.5
assert result['vlm_gradient_norm']>0
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
atomic(a.out,result);print(json.dumps(result),flush=True)
