"""Training-corpus visual features before patch vocabulary quantization.

These are permitted neural boundary inputs, not continuous dynamics targets.
No pose, distance, map, collision or reward data are loaded.
"""
import json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from predictive_nesy.environment import FrozenVision
from predictive_nesy.data import sha

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'prequant_visual_features_v18';assert not out.exists()
assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}[7]<300
torch.set_num_threads(2);started=time.time()
source=root/'run_definition_world_v12/checkpoint_001600.pt'
assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
saved=torch.load(source,map_location='cpu',weights_only=False);manifest=saved['manifest'];directory=Path(manifest['data']['source'])
rows=json.loads((directory/'training_rows.json').read_text());tokens=np.load(directory/'visual_tokens.npy')
assert sha(directory/'training_rows.json')==manifest['data']['rows_sha256']
assert sha(directory/'visual_tokens.npy')==manifest['data']['codes_sha256']
vision=FrozenVision(manifest['frozen_vision'])
from nesy_vla.runtime import image_tensor
v=vision.vocabulary;dimensions=v['axes'].shape[-1];out.mkdir()
features=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float16,shape=(len(rows),49,dimensions))
def load(row):
    with Image.open(row['path']) as image:return image_tensor(image.convert('RGB'),vision.cfg['image_size'],'cpu')[0]
match=0;total=0
with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
    for begin in range(0,len(rows),32):
        x=torch.stack(list(pool.map(load,rows[begin:begin+32]))).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            spatial=F.adaptive_avg_pool2d(vision.vision(x),(7,7)).flatten(2).transpose(1,2)
        z=F.normalize(((spatial.float()-v['mean'])/v['sd'])@v['axes'],dim=-1)
        features[begin:begin+len(x)]=z.cpu().numpy().astype(np.float16)
        ids=torch.cdist(z.flatten(0,1),v['centers']).argmin(1).reshape(len(x),49).cpu().numpy()
        match+=int((ids==tokens[begin:begin+len(x)]).sum());total+=ids.size
        if begin%2048==0:
            features.flush();print(json.dumps(dict(event='feature_cache',completed=begin+len(x),total=len(rows),seconds=time.time()-started)),flush=True)
features.flush();del features
result=dict(training_only=True,source=str(directory),source_checkpoint=str(source),source_sha256=sha(source),
    rows=len(rows),patches=49,features_per_patch=dimensions,dtype='float16',frozen_vision=manifest['frozen_vision'],
    row_manifest_sha256=sha(directory/'training_rows.json'),source_token_sha256=sha(directory/'visual_tokens.npy'),
    feature_sha256=sha(out/'features.npy'),batched_nearest_token_agreement=match/total,
    scope='Before learned patch-vocabulary argmax; neural input only, no continuous transition target',
    seconds=time.time()-started)
(out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
