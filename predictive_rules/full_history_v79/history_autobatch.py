import gc,time,statistics
import torch
import torch.distributed as dist
from torch.nn import functional as F
def choose(predictor,data,rng,limit_gib=21.):
    rows=[];device=data.device
    for batch in (16384,32768,65536,131072):
        predictor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        samples=[];inputs=lp=loss=None;okay=True
        try:
            for repeat in range(3):
                predictor.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter()
                ids=data.sample(rng,batch);inputs=data.inputs(ids);lp=predictor(**inputs)
                loss=F.nll_loss(lp,data.columns['target'][ids])+.2*F.kl_div(lp,inputs['base_logp'].softmax(-1),reduction='batchmean')
                loss.backward();torch.cuda.synchronize();assert torch.isfinite(loss)
                if repeat:samples.append(time.perf_counter()-tick)
                inputs=lp=loss=None
        except torch.OutOfMemoryError:okay=False;inputs=lp=loss=None
        predictor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        valid=torch.tensor(float(okay),device=device);dist.all_reduce(valid,op=dist.ReduceOp.MIN)
        values=torch.tensor([torch.cuda.max_memory_allocated()/2**30,statistics.median(samples) if okay else 1e9],device=device)
        dist.all_reduce(values,op=dist.ReduceOp.MAX);peak,seconds=values.tolist()
        rows.append(dict(batch=batch,okay=bool(valid.item()),peak_gib=peak,seconds=seconds,rows_per_second=batch/seconds))
        if not valid.item() or peak>limit_gib:break
    chosen=max((r for r in rows if r['okay'] and r['peak_gib']<=limit_gib),key=lambda r:r['rows_per_second'])
    torch.cuda.reset_peak_memory_stats();return chosen['batch'],dict(candidates=rows,chosen=chosen,optimizer_updates_in_capacity_test=0)
