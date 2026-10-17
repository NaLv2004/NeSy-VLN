"""Encode verified recovery images with exactly the frozen v43 vocabulary."""
import gc,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'data/onpolicy_recovery_v48';sys.path.insert(0,str(b/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,setup,load_checkpoint,image_tensor,atomic
from nesy_vla.model import NavigationModel
torch.set_num_threads(2);torch.manual_seed(4300)
complete=read(out/'COMPLETE');assert sha(out/'training_rows.json')==complete['training_rows_sha256']
source=b/'spatial_automaton_v43';commit=read(source/'COMMITTED');assert sha(source/'visual_vocabulary.pt')==commit['vocabulary_sha256']
if (out/'TOKENS_COMMITTED').exists():raise SystemExit('Already encoded; retain original codes')
vision_source=b/'symbolic_conjunctive_v23/runs/conjunctive/best_development';cfg=read(vision_source.parent/'config.json');setup(cfg)
model=NavigationModel(cfg).eval();load_checkpoint(vision_source,model);model.requires_grad_(False).to('cuda:0');vision=model.grounder.vision;del model;gc.collect();v=torch.load(source/'visual_vocabulary.pt',map_location='cuda:0',weights_only=True)
@torch.no_grad()
def encode(image):
 x=image_tensor(image,cfg['image_size'],'cuda:0')
 with torch.autocast('cuda',dtype=torch.bfloat16):f=torch.nn.functional.adaptive_avg_pool2d(vision(x),(7,7)).flatten(2).transpose(1,2)
 z=torch.nn.functional.normalize(((f.float()-v['mean'])/v['sd'])@v['axes'],dim=-1)
 return torch.cdist(z.flatten(0,1),v['centers']).argmin(1).cpu().numpy().astype(np.int16)
for p in sorted((source/'development').glob('*/trace.json'))[:3]:
 t=read(p);assert encode(Image.open(p.parent/'first.png').convert('RGB')).tolist()==t['steps'][0]['symbols'][:49]
rows=read(out/'training_rows.json');codes=np.empty((len(rows),49),dtype=np.int16)
for i,r in enumerate(rows):
 assert sha(r['path'])==r['image_sha256'];codes[i]=encode(Image.open(r['path']).convert('RGB'))
 if i%256==0:atomic(out/'encoding_status.json',{'state':'encoding','rows':i+1,'total':len(rows),'time':time.time()})
np.save(out/'visual_tokens.npy',codes);atomic(out/'TOKENS_COMMITTED',{'time':time.time(),'rows':len(rows),'tokens_sha256':sha(out/'visual_tokens.npy'),'training_rows_sha256':complete['training_rows_sha256'],'vocabulary_sha256':commit['vocabulary_sha256'],'encoding':'Identicaloneimagepath; vocabulary not refitted. No heldout image fitted.'})
atomic(out/'encoding_status.json',{'state':'complete','rows':len(rows),'time':time.time()});print(json.dumps(read(out/'TOKENS_COMMITTED'),indent=2))
