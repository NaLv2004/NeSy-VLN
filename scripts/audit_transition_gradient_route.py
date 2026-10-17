"""Training-only gradient attribution on a frozen, actually evaluated checkpoint."""
import json,os,sys,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_robust_v7'
sys.path.insert(0,str(root));os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstrations_many
from nesy_vla.losses import transition_loss

torch.set_num_threads(2);torch.manual_seed(14)
cfg=read(root/'configs/robust_multieffect.json');setup(cfg);model=NavigationModel(cfg)
checkpoint=root/'runs/robust_multieffect/evaluations/step_000010/checkpoint'
load_checkpoint(checkpoint,model);bank=DataBank(cfg)
episode=next(e for e in sorted(bank.demo_ids) if len(bank.demos[e])>=5)
bank.demos[episode]=bank.demos[episode][:3];model.eval();policy=demonstrations_many(model,bank,[episode])[0]
chunk=policy.chunks[0];rows=chunk['rows'][:2];model.train()
program=model.generator(chunk['task'],chunk['images'],chunk['feedback'],chunk['previous'])
memory=[];prior=None;bc=[];direct=[];detached=[]
for row in rows:
    result=model.step(program,row['image'],memory,prior)
    bc.append(-result['probs'][row['action']].log())
    prior,memory,_=model.advance(program,result,row['action'],row['feedback'],memory)
    target,confidence=model.transition_target(program,row['next_image'],memory)
    direct.append(transition_loss(prior[0],target,result['belief'],confidence,balanced=True))
    prediction,_,_=model.logic.transition(program,result['belief'].detach(),result['confidence'].detach(),row['action'],
        model.grounder.retain(program.definitions,row['action']))
    detached.append(transition_loss(prediction,target,result['belief'],confidence,balanced=True))
assert torch.allclose(torch.stack(direct),torch.stack(detached),atol=1e-6),'Detach must preserve every forward value'
params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
terms={'bc':torch.stack(bc).mean(),'transition_direct':torch.stack(direct).mean()*cfg['loss']['transition'],
       'transition_detached_source':torch.stack(detached).mean()*cfg['loss']['transition']}
gradients={key:torch.autograd.grad(loss,[p for _,p in params],retain_graph=True,allow_unused=True) for key,loss in terms.items()}
prefixes=['grounder.vision.','grounder.query.','grounder.retention.','generator.head.effect.','generator.head.premise.','logic.']
metrics={}
for name,grads in gradients.items():
    metrics[name]={}
    for prefix in prefixes:
        squared=dot=bc_squared=0.
        for (param_name,param),grad,bc_grad in zip(params,grads,gradients['bc']):
            if not param_name.startswith(prefix):continue
            if grad is not None:squared+=float(grad.square().sum())
            if bc_grad is not None:bc_squared+=float(bc_grad.square().sum())
            if grad is not None and bc_grad is not None:dot+=float((grad*bc_grad).sum())
        metrics[name][prefix]={'norm':squared**.5,'cosine_with_bc':dot/max(1e-15,(squared*bc_squared)**.5)}
result={'time':time.time(),'checkpoint':str(checkpoint),'episode_id':episode,'training_frames':2,
        'forward_losses':{k:float(v.detach()) for k,v in terms.items()},'gradient_attribution':metrics,
        'qualification':'Frozen-checkpoint gradient diagnostic on two actual training transitions; not navigation performance or a deployed change.'}
atomic(base/'diagnostics_20261011/transition_gradient_route.json',result);print(json.dumps(result,indent=2))
