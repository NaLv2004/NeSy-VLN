import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nesy_vla.runtime import read,setup,DataBank,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstration
from nesy_vla.training import train_batch

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/navigation.json');p.add_argument('--out',required=True)
p.add_argument('--batches',type=int,nargs='+',default=[2,4,8,12,16,20,24]);a=p.parse_args()
cfg=read(a.config);setup(cfg);torch.cuda.set_device(0);bank=DataBank(cfg)
torch.manual_seed(cfg['seed']);model=NavigationModel(cfg).cuda().eval()
longest=sorted(bank.demo_ids,key=lambda e:len(bank.demos[e]),reverse=True)[0]
demo=demonstration(model,bank,longest)
chunks=[c for c in demo.chunks if c['previous'] is not None and len(c['rows'])==10]
assert chunks,'Need a real ten-action revision chunk'
chunk=max(chunks,key=lambda c:c['start'])
params=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(params,lr=1e-5);results=[]
for size in a.batches:
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model.train();start=time.time()
    try:
        optimizer.zero_grad(set_to_none=True)
        loss,stats=train_batch(model,[(chunk,True)]*size)
        loss.backward();torch.nn.utils.clip_grad_norm_(params,2.);optimizer.step();torch.cuda.synchronize()
        row={'batch':size,'seconds':time.time()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'training_frames':size*10,'ok':True,
            'loss':float(loss.detach())}
        del loss,stats
    except torch.cuda.OutOfMemoryError:
        row={'batch':size,'ok':False,'oom':True,'seconds':time.time()-start}
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    results.append(row);atomic(a.out,{'results':results,'episode_id':longest,'revision_at':chunk['start'],
        'includes':'real VLM revision + ten-step BPTT + EMA + BC/outcome/transition/structure backward + AdamW',
        'total_memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30})
    print(json.dumps(row),flush=True)
    if not row['ok'] or row.get('peak_reserved_gib',0)>23:break
