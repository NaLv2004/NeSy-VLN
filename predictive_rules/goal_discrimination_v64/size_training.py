"""Short compute sizing, no committed model changes or navigation trials."""
import gc,json,os,random,time
from pathlib import Path
import torch
from model import FactoredWorld,Config
from data import Corpus,ROOT,sha
from feedback_objective import FeedbackData,FeedbackObjective
from accelerated_transition import install
from train_feedback import exclude_feedback_holdout

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    data=ROOT/'goal_discrimination_v64';p=data/'training_capacity.json'
    assert not p.exists(),'Do not repeat a completed benchmark'
    saved=torch.load(data/'source.pt',map_location='cpu',weights_only=False)
    measured=FeedbackData(data,saved['manifest']['transition_holdout_episodes'],'cuda')
    corpus=Corpus(0,2,str(ROOT/'onpolicy_successors_v60'));exclude_feedback_holdout(corpus,measured.holdout)
    model=FactoredWorld(Config(**saved['model_config'])).cuda();model.load_state_dict(saved['model']);install(model);del saved
    wrapper=FeedbackObjective(model);rng=random.Random(641017);rows=[]
    for sizes in ((128,64,256),(256,128,512),(384,192,768)):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
        try:
            base=corpus.sample(rng,sizes[0],sizes[1],'cuda');base['world_updates_encoder']=True
            measured_batch=measured.sample(rng,sizes[2])
            for iteration in range(4):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.time()
                loss,metrics=wrapper(base,measured_batch);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                assert torch.isfinite(loss) and torch.isfinite(norm)
                missing=[n for n,v in model.named_parameters() if v.requires_grad and v.grad is None]
                assert not missing,missing
                torch.cuda.synchronize();elapsed=time.time()-tick
                if iteration:times.append(elapsed)
            mean=sum(times)/len(times);peak=torch.cuda.max_memory_allocated()/2**30
            rows.append(dict(batch=list(sizes),seconds=mean,peak_gib=peak,
                relative_examples_per_second=sizes[0]/mean,finite_backward=True,loss=float(loss),gradient_norm=float(norm)))
            print(json.dumps(rows[-1]),flush=True)
            del base,measured_batch,loss,metrics,norm;model.zero_grad(set_to_none=True);gc.collect()
            if peak>20:break
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(batch=list(sizes),oom=True));break
    candidates=[r for r in rows if r.get('finite_backward') and r['peak_gib']<=20]
    assert candidates,rows
    best=max(candidates,key=lambda r:r['relative_examples_per_second'])
    result=dict(time=time.time(),rows=rows,selected_batch=best['batch'],peak_limit_gib=20,
        verified_finite_backward=True,optimizer_steps=0,model_saved=False,
        claim='Compute-only sizing; formal DDP throughput reported separately')
    p.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
