"""Small same-data performance checks; does not update any checkpoint."""
import gc,json,os,statistics,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config
from accelerated_transition import install
from joint_data import JointData
from joint_objective import JointObjective,configure
from data import ROOT
from planner import Program as CPUProgram,plan
from gpu_planner import Program,extend_batch

OUT=ROOT/'gpu_efficiency_v74';OUT.mkdir(exist_ok=True)
def save(row):
    tmp=OUT/'benchmark.tmp';tmp.write_text(json.dumps(row,indent=2));tmp.replace(OUT/'benchmark.json')

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2';torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=True;device=torch.device('cuda')
    saved=torch.load(ROOT/'parallel_prefix_v72/source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model']);install(model)
    data=JointData(model,saved['manifest']['transition_holdout_episodes'],device)
    report=dict(state='benchmarking',pid=os.getpid(),time=time.time(),optimizer_updates=0,batches=[],planner=[])
    save(report)
    exported=[]
    with torch.no_grad():
        for i in range(8):
            exported.append(model.compile(data.context[i:i+1],data.task[i:i+1]))
    for batch in (1,8):
        cp=[CPUProgram(**x).validate() for x in exported[:batch]]
        gp=[Program(**x).validate() for x in exported[:batch]]
        started=time.perf_counter()
        for p in cp:p.extend(32)
        cpu_seconds=time.perf_counter()-started
        extend_batch(gp,32);torch.cuda.synchronize()
        gp=[Program(**x).validate() for x in exported[:batch]]
        started=time.perf_counter();extend_batch(gp,32);torch.cuda.synchronize();gpu_seconds=time.perf_counter()-started
        differences=total=0;maxerror=0;root_disagreements=[]
        for i,(a,b) in enumerate(zip(cp,gp)):
            for h in range(1,33):
                different=a._policies[h]!=b._policies[h];differences+=int(different.sum());total+=len(different)
                maxerror=max(maxerror,float(np.abs(a._levels[h]-b._levels[h]).max()),float(np.abs(a._failures[h]-b._failures[h]).max()))
            for m,o in [(0,0),(1,1),(17,31),(31,63)]:
                pa=plan(a,m,o);pb=plan(b,m,o)
                if pa.get('actions',[None])[:1]!=pb.get('actions',[None])[:1]:root_disagreements.append([i,m,o])
        report['planner'].append(dict(programs=batch,horizon=32,cpu_seconds=cpu_seconds,gpu_seconds=gpu_seconds,
            speedup=cpu_seconds/gpu_seconds,all_state_policy_differences=differences,all_state_decisions=total,
            max_probability_error=maxerror,probe_root_action_disagreements=root_disagreements))
        assert maxerror<1e-10
        save(report)
    del cp,gp,exported
    configure(model);objective=JointObjective(model,data);rng=torch.Generator(device=device).manual_seed(74017)
    for feedback,pairs in [(8192,1024),(16384,2048),(16384,4096),(32768,6144)]:
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        times=[];loss=metrics=None
        try:
            for repeat in range(4):
                f,p=data.sample(rng,feedback,pairs);model.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter()
                loss,metrics=objective(f,p);loss.backward();torch.cuda.synchronize()
                assert torch.isfinite(loss)
                if repeat:times.append(time.perf_counter()-tick)
                del loss,metrics;loss=metrics=None
            seconds=statistics.median(times);peak=torch.cuda.max_memory_allocated()/2**30
            report['batches'].append(dict(feedback=feedback,transitions=pairs,seconds=seconds,
                transition_rows_per_second=pairs/seconds,feedback_rows_per_second=feedback/seconds,peak_gib=peak,state='ok'))
        except torch.OutOfMemoryError:
            report['batches'].append(dict(feedback=feedback,transitions=pairs,state='oom'))
            loss=metrics=None;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        save(report)
    choices=[r for r in report['batches'] if r['state']=='ok' and r['peak_gib']<21]
    report['recommended_batch']=max(choices,key=lambda r:r['transition_rows_per_second'])
    report.update(state='complete',time=time.time());save(report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
