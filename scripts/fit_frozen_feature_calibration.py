"""Fit only per-channel visual statistics on the locked training split."""
import os,sys,time,json,random,hashlib
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/calibrated_vision_v10')
sys.path.insert(0,str(root));os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
from PIL import Image
from nesy_vla.runtime import read,setup,DataBank,image_tensor,atomic
from nesy_vla.vision_backbone import FrozenQwenVision
torch.set_num_threads(2);torch.manual_seed(919);torch.cuda.set_per_process_memory_fraction(.16)
cfg=read('/home/user/xjy/nesy_vla_20261010/symbolic_vision_v8/configs/frozen_qwen.json');setup(cfg);bank=DataBank(cfg)
all_rows=[r for rows in bank.demos.values() for r in rows]
selected=random.Random(919).sample(all_rows,min(512,len(all_rows)))
vision=FrozenQwenVision(cfg['model_path']).to('cuda').eval();sums=squares=None;count=0;start=time.time()
with torch.no_grad():
    for begin in range(0,len(selected),8):
        images=[Image.open(bank.demo_image_path(r)).convert('RGB') for r in selected[begin:begin+8]]
        values=vision(torch.cat([image_tensor(im,cfg['image_size'],'cuda') for im in images])).double()
        batch_sum=values.sum((0,2,3));batch_square=values.square().sum((0,2,3))
        sums=batch_sum if sums is None else sums+batch_sum;squares=batch_square if squares is None else squares+batch_square
        count+=values.shape[0]*values.shape[2]*values.shape[3]
        if begin%64==0:print(json.dumps({'encoded_training_frames':begin+len(images),'seconds':time.time()-start}),flush=True)
mean=sums/count;std=(squares/count-mean.square()).clamp_min(0).sqrt();floor=float(std.median())*.1
state={'mean':mean.float().cpu(),'scale':std.clamp_min(max(floor,1e-5)).float().cpu()}
path=root/'training_feature_calibration.pt';torch.save(state,path)
atomic(root/'calibration_provenance.json',{'time':time.time(),'training_only':True,'label_free':True,'data':bank.provenance,
    'frames':[{'episode_id':r['episode_id'],'image':r['image']} for r in selected],
    'feature_vectors':count,'scale_floor':floor,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'encoder_path':cfg['model_path'],'size':cfg['image_size'],'seconds':time.time()-start})
print(json.dumps({'saved':str(path),'training_frames':len(selected),'seconds':time.time()-start}),flush=True)
