"""Compare cached training evidence with the actual runtime CUDA frontend."""
import json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_contextual_symbols_v48'))
from model import FactoredWorld,Config,hard
from predictive_nesy.environment import FrozenVision
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
saved=torch.load(ROOT/'diagnostic_goal_aware_symbols_v52/source.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'])
manifest=saved['manifest'];del saved
rows=json.loads((Path(manifest['data']['source'])/'training_rows.json').read_text())
cache=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
dataset=json.loads((ROOT/'diagnostic_dinov2_v43/dataset.json').read_text())
chosen=[]
for split in ('fit','heldout'):
    selected=list(dict.fromkeys(r['current'] for r in dataset['records'] if (r['split']=='fit')==(split=='fit')))
    chosen.extend(selected[:16])
frontend=FrozenVision(manifest['frozen_vision']);gate=NeuralBoundaryGate([frontend.vision,model],interval=1)
encoder=RawBoundaryGraphEncoder(frontend,gate);values=[]
with torch.inference_mode():
    for step,i in enumerate(chosen):
        with Image.open(rows[i]['path']) as image:image=image.convert('RGB')
        with gate.boundary(step):
            _,runtime=encoder.encode_joint([image]);training=torch.tensor(cache[i:i+1].copy(),device='cuda')
            task=torch.tensor([['chair','table'].index(rows[i]['category'])],device='cuda')
            def context(x):return model.context_sequence(x[:,None],hard(model.observe(x))[:,None],torch.tensor([[17]],device='cuda'),task)[:,0]
            tc=context(training);rc=context(runtime)
            a=model.observe(training,tc);b=model.observe(runtime,tc);c=model.observe(runtime,rc)
            values.append(dict(row=i,shape=list(image.size),feature_cosine=float(F.cosine_similarity(training.float(),runtime.float(),dim=-1).mean()),
                feature_rms=float((training.float()-runtime.float()).square().mean().sqrt()),
                max_feature_error=float((training.float()-runtime.float()).abs().max()),
                cache_code=int(a.argmax()),runtime_fixed_context_code=int(b.argmax()),runtime_context_code=int(c.argmax()),
                max_logit_error=float((a-b).abs().max()),context_rms=float((tc-rc).square().mean().sqrt())))
gate.close()
report=dict(time=time.time(),samples=len(values),training_only=True,no_new_rollouts=True,
    fixed_context_code_agreement=sum(r['cache_code']==r['runtime_fixed_context_code'] for r in values)/len(values),
    actual_context_code_agreement=sum(r['cache_code']==r['runtime_context_code'] for r in values)/len(values),
    mean_feature_cosine=sum(r['feature_cosine'] for r in values)/len(values),rows=values)
out=ROOT/'diagnostic_rl_perception_v53/frontend_equivalence.json';out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
