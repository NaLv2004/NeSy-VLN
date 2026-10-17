"""Actual pretrained vision-only inference and symbolic training smoke check."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from PIL import Image
from nesy_vla.vision_backbone import FrozenQwenVision
from nesy_vla.runtime import read,atomic,image_tensor,setup,DataBank,initialize_symbolic_modules
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstrations_many
from nesy_vla.training import train_batch

root=Path('/home/user/xjy/nesy_vla_20261010');code=root/'frozen_vision_v8_code';torch.set_num_threads(2)
cfg=read(root/'symbolic_robust_v7/configs/robust_multieffect.json')
cfg.update(vision_backbone='qwen35_frozen',freeze_vision=True,robust_grounding=False,vision_microbatch=4)
setup(cfg);started=time.time();model=NavigationModel(cfg).eval();initialization=initialize_symbolic_modules(cfg['initialize_from'],model)
bank=DataBank(cfg);episode=next(e for e in sorted(bank.demo_ids) if len(bank.demos[e])>=3)
# Three actual training frames suffice to verify the forward/backward boundary.
bank.demos[episode]=bank.demos[episode][:3]
with torch.no_grad():policies=demonstrations_many(model,bank,[episode])
model.train();loss,stats=train_batch(model,[(policies[0].chunks[0],True)]);loss.backward()
gradients={prefix:sum(float(p.grad.square().sum()) for n,p in model.named_parameters()
    if p.grad is not None and n.startswith(prefix))**.5 for prefix in ['grounder.vision.','grounder.projection.','grounder.query.','generator.head.','logic.']}
assert gradients['grounder.vision.']==0 and all(v>0 for k,v in gradients.items() if k!='grounder.vision.')
assert model.teacher.vision is model.grounder.vision
result={'time':time.time(),'seconds':time.time()-started,'initialization':initialization,
    'episode_id':episode,'frames':3,'loss':float(loss.detach()),'gradients':gradients,'metrics':stats,
    'vision_parameters':sum(p.numel() for p in model.grounder.vision.parameters()),
    'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
    'only_vision_pretrained_weights_loaded':True,'language_action_decoder_loaded':False,
    'all_actions_via_discrete_symbols_and_generated_rules':True,
    'qualification':'Execution and gradient check; not a navigation evaluation.'}
atomic(code/'pretrained_vision_verification.json',result);print(json.dumps(result,indent=2))
