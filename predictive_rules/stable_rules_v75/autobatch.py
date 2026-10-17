"""Measure actual forward/backward throughput before committing any update."""
import gc,time,statistics
import torch
import torch.distributed as dist

def choose(model,objective,data,rng,limit_gib=21.0):
    rows=[];device=next(model.parameters()).device
    for feedback,pairs in ((16384,2048),(16384,4096),(32768,6144),(32768,8192),(32768,12288)):
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        samples=[];loss=metrics=None;okay=True
        try:
            for repeat in range(3):
                f,p=data.sample(rng,feedback,pairs);model.zero_grad(set_to_none=True)
                torch.cuda.synchronize();tick=time.perf_counter()
                loss,metrics=objective(f,p);loss.backward();torch.cuda.synchronize()
                assert torch.isfinite(loss)
                if repeat:samples.append(time.perf_counter()-tick)
                del loss,metrics;loss=metrics=None
        except torch.OutOfMemoryError:
            okay=False;loss=metrics=None
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        local=torch.tensor([float(okay),torch.cuda.max_memory_allocated()/2**30,
            statistics.median(samples) if okay else 1e9],device=device,dtype=torch.float64)
        valid=local[:1].clone();dist.all_reduce(valid,op=dist.ReduceOp.MIN)
        dist.all_reduce(local[1:],op=dist.ReduceOp.MAX)
        ok=bool(valid.item());peak=float(local[1]);seconds=float(local[2])
        rows.append(dict(feedback=feedback,transitions=pairs,peak_gib=peak,seconds=seconds,okay=ok,
                         transition_rows_per_second=pairs/seconds if ok else 0.))
        if not ok or peak>limit_gib:break
    eligible=[r for r in rows if r['okay'] and r['peak_gib']<=limit_gib]
    assert eligible,rows
    chosen=max(eligible,key=lambda r:r['transition_rows_per_second'])
    torch.cuda.reset_peak_memory_stats()
    return chosen,dict(candidates=rows,chosen=chosen,limit_gib=limit_gib,optimizer_updates_in_capacity_test=0,
                       selection='Maximum measured transition rows/sec below memory budget on both GPUs')
