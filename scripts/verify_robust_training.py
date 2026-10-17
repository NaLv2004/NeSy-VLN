"""A real training-data forward/backward check before reallocating GPUs."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from nesy_vla.runtime import setup,DataBank,read,load_checkpoint,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstrations_many
from nesy_vla.training import train_batch

root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_robust_v7')
records=[]
for name in ['robust_reference','robust_multieffect']:
    cfg=read(root/'configs'/f'{name}.json');setup(cfg)
    # The final lossless dataset is verified separately before training starts.
    cfg.pop('demonstration_file');cfg.pop('demonstration_image_root')
    bank=DataBank(cfg);torch.manual_seed(46)
    model=NavigationModel(cfg).eval();state=load_checkpoint(cfg['initialize_from'],model)
    chosen=next(x for x in sorted(bank.demo_ids) if len(bank.demos[x])>=5)
    policies=demonstrations_many(model,bank,[chosen])
    chunk=policies[0].chunks[0];chunk['rows']=chunk['rows'][:3]
    model.train();loss,stats=train_batch(model,[(chunk,True)]);loss.backward()
    gradients={prefix:sum(float(p.grad.square().sum()) for n,p in model.named_parameters()
        if p.grad is not None and n.startswith(prefix))**.5
        for prefix in ['generator.head.effect.','generator.head.premise.','grounder.query.','grounder.vision.','logic.']}
    assert torch.isfinite(loss) and all(value>0 for value in gradients.values())
    assert stats[0]['visual_consistency']>=0 and 'view_bit_disagreement_metric' in stats[0]
    program=model.generator(chunk['task'],chunk['images'],chunk['feedback'],chunk['previous']);program.validate()
    result={'arm':name,'source_update':state['update'],'episode_id':chosen,'frames':3,
        'loss':float(loss.detach()),'gradients':gradients,'metrics':stats[0],'effect_mode':program.effect_mode}
    records.append(result);print(json.dumps(result),flush=True)
    del model,loss,policies,program
atomic(root/'training_verification.json',{'time':time.time(),'arms':records,'qualification':'Training execution check, not navigation success.'})
