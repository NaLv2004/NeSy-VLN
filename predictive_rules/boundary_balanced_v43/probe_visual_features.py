"""Train-only, episode-disjoint inverse-dynamics probe of frozen visual features.

This diagnostic head never enters the symbolic agent or chooses runtime actions.
Both encoders use the same complete RGB frames resized to 224x224 and the same
7x7x128 feature budget. DINO channel projection is fitted on probe-fit episodes.
No navigation validation examples, coordinates, collisions or semantics loaded.
"""
import hashlib,json,os,random,re,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Dinov2Model

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_dinov2_v43'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.20)
torch.backends.cuda.matmul.allow_tf32=True
torch.manual_seed(431017);rng=random.Random(431017);began=time.time()
OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'run_action_grounded_v33/manifest.json').read_text())
source=Path(manifest['data']['source']);row_file=source/'training_rows.json'
assert hashlib.sha256(row_file.read_bytes()).hexdigest()==manifest['data']['rows_sha256']
rows=json.loads(row_file.read_text());episodes=defaultdict(list);canonical={}
for i,row in enumerate(rows):
    name=row['episode_id'];origin=row.get('source_episode_id',name)
    layout='layout_'+re.search(r'P25_train_(\d+)_',origin)[1]
    assert layout in manifest['data']['train_layouts']
    canonical[name]=origin
    episodes[name].append(i)
representative={}
for name in sorted(episodes):representative.setdefault(canonical[name],name)
names=sorted(representative.values());rng.shuffle(names)
assert len(names)>=384
fit_names=names[:256];test_names=names[256:384]
assert not set(fit_names)&set(test_names)
assert not {canonical[n] for n in fit_names}&{canonical[n] for n in test_names}
records=[]
for split,names in [('fit',fit_names),('heldout_train_episodes',test_names)]:
    for name in names:
        indices=episodes[name];n=len(indices)
        positions=sorted(set([0,n-1]+rng.sample(range(n),min(n,6))))
        for t in positions:
            if not rows[indices[t]].get('supervised',True):continue
            records.append(dict(split=split,episode=name,current=indices[t],
                successor=indices[min(t+1,n-1)],action=int(rows[indices[t]]['label'])))
indices=sorted({r[key] for r in records for key in ('current','successor')});lookup={v:i for i,v in enumerate(indices)}
fit_rows=sorted({lookup[r[key]] for r in records if r['split']=='fit' for key in ('current','successor')})
dataset=dict(training_only=True,fit_episode_ids=fit_names,heldout_episode_ids=test_names,
    records=records,source_row_indices=indices,seed=431017,source_sha256=manifest['data']['rows_sha256'])
(OUT/'dataset.json').write_text(json.dumps(dataset))
asset=ROOT/'assets/dinov2_small';provenance=json.loads((asset/'provenance.json').read_text())
for record in provenance['files']:
    assert hashlib.sha256((asset/record['name']).read_bytes()).hexdigest()==record['sha256']
encoder=Dinov2Model.from_pretrained(str(asset),local_files_only=True).eval().cuda()
mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
def load(i):
    with Image.open(rows[i]['path']) as im:
        rgb=np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).copy()
    return (torch.from_numpy(rgb).permute(2,0,1).float()/255-mean)/std
features=[]
with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
    for begin in range(0,len(indices),32):
        image=torch.stack(list(pool.map(load,indices[begin:begin+32]))).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=encoder(pixel_values=image).last_hidden_state[:,1:]
            assert hidden.shape[1]==256
            patch=F.adaptive_avg_pool2d(hidden.transpose(1,2).reshape(-1,384,16,16),(7,7)).flatten(2).transpose(1,2)
        features.append(patch.float().cpu())
        if begin%512==0:print(json.dumps(dict(stage='encoding',images=min(begin+32,len(indices)),total=len(indices),seconds=time.time()-began)),flush=True)
dino=torch.cat(features);del encoder,features;torch.cuda.empty_cache()
patch=dino[fit_rows].reshape(-1,384)
if len(patch)>60000:patch=patch[torch.randperm(len(patch))[:60000]]
mu=patch.mean(0);sd=patch.std(0).clamp_min(1e-4);standard=(patch-mu)/sd
_,axes=torch.linalg.eigh(standard.T@standard/max(1,len(standard)-1));axes=axes[:,-128:]
dino=F.normalize(((dino-mu)/sd)@axes,dim=-1)
projection=dict(mean=mu,sd=sd,axes=axes,fit_episode_ids=fit_names,model=provenance)
torch.save(projection,OUT/'training_only_projection.pt')
np.save(OUT/'dinov2_features.npy',dino.numpy().astype(np.float16))
raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
conv=torch.from_numpy(raw[indices].astype(np.float32))
labels=torch.tensor([r['action'] for r in records],device='cuda')
fit=torch.tensor([r['split']=='fit' for r in records],device='cuda')
current=torch.tensor([lookup[r['current']] for r in records]);future=torch.tensor([lookup[r['successor']] for r in records])
results={}
def scores(logits,mask):
    correct=(logits.argmax(-1)==labels);out=dict(accuracy=float(correct[mask].float().mean()),samples=int(mask.sum()))
    by_action={str(a):dict(n=int(((labels==a)&mask).sum()),accuracy=float(correct[(labels==a)&mask].float().mean()))
        for a in range(17) if ((labels==a)&mask).any()}
    out.update(by_opaque_action_id=by_action,macro_accuracy=sum(v['accuracy'] for v in by_action.values())/len(by_action))
    return out
for label,features in [('existing_convnext',conv),('dinov2_small',dino)]:
    torch.manual_seed(431017)
    before=features[current].flatten(1);after=features[future].flatten(1)
    values=torch.cat([before,after-before],-1).cuda()
    mu=values[fit].mean(0);sd=values[fit].std(0).clamp_min(.01);values=(values-mu)/sd
    head=nn.Sequential(nn.Linear(values.shape[-1],128),nn.SiLU(),nn.Linear(128,17)).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
    fit_indices=fit.nonzero()[:,0];tick=time.time()
    for update in range(500):
        batch=fit_indices[torch.randint(len(fit_indices),(min(256,len(fit_indices)),),device='cuda')]
        loss=F.cross_entropy(head(values[batch]),labels[batch]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    with torch.no_grad():
        logits=head(values);result=dict(fit=scores(logits,fit),heldout_train_episodes=scores(logits,~fit),seconds=time.time()-tick)
    results[label]=result
    print(json.dumps(dict(stage='probe_result',encoder=label,**result)),flush=True)
    del head,optimizer,values;torch.cuda.empty_cache()
baseline_action=int(torch.bincount(labels[fit],minlength=17).argmax())
report=dict(training_only=True,frame_count=len(indices),fit_episodes=len(fit_names),heldout_episodes=len(test_names),
    results=results,majority_action=baseline_action,heldout_majority_accuracy=float((labels[~fit]==baseline_action).float().mean()),
    seconds=time.time()-began,memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,
    runtime_action_head_installed=False,limitations='Inverse dynamics evidence only; no claim of better discrete multi-step prediction or navigation; one diagnostic split and seed. Episode holdout applies to the probe head and DINO PCA only. The existing ConvNeXt encoder was previously trained on this broader corpus; this is a drop-in comparison, not a clean comparison of unseen-domain pretraining.',
    image_geometry='Full RGB resized224x224, ImageNet normalization,49patches x128channels for both')
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
