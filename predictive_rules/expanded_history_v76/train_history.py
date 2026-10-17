"""A single history-predictor experiment on GPUs 4/5; frozen perception."""
import json,os,time,gc
import torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from model import FactoredWorld,Config
from data import ROOT,sha
from joint_data import JointData
from history_data import HistoryData
from history_model import HistoryTransition
from accelerated_transition import install
from history_autobatch import choose
from evaluate import atomic,evaluate,summarize

OUT=ROOT/'run_expanded_history_v76'

def log(name,row):
    with (OUT/name).open('a') as stream:stream.write(json.dumps(row)+'\n')

def main():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5' and int(os.environ['WORLD_SIZE'])==2
    torch.cuda.set_device(local);torch.set_num_threads(3);torch.manual_seed(691017)
    torch.backends.cuda.matmul.allow_tf32=True
    dist.init_process_group('nccl');device=torch.device('cuda',local)
    source=ROOT/'run_goal_probability_v68';commit=json.loads((source/'best.COMMITTED.json').read_text())
    path=source/commit['filename'];assert sha(path)==commit['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved['model']);install(model)
    saved_mean=model.evidence_mean.clone();saved_scale=model.evidence_scale.clone()
    data=JointData(model,saved['manifest']['transition_holdout_episodes'],device)
    assert torch.equal(model.evidence_mean,saved_mean) and torch.equal(model.evidence_scale,saved_scale)
    data=HistoryData(model,data)
    predictor=HistoryTransition().to(device)
    optimizer=torch.optim.AdamW(predictor.parameters(),lr=2e-4,weight_decay=.01)
    rng=torch.Generator(device=device).manual_seed(691017+rank)
    manifest=dict(saved['manifest']);del saved
    manifest.update(architecture='expanded_history_v76',history_source_checkpoint=dict(path=str(path),**commit),
        predictor_inputs='Last2 discrete observation IDs, historical action IDs, current discrete memory, candidate action, generated definitions without raw program context',
        predictor_outputs='Distribution over next64anonymous observation IDs only; no action/value/reward output',
        inference='One history-conditioned next-state expansion followed by unchanged symbolic table Bellman continuation; memory still regrounded only at program renewal',
        fixed_perception_and_definitions=True,gpu_symbolic_planner=True,validation_workers_per_gpu=40,source_anchor_kl_weight=.2,maximum_residual_logit=2,raw_context_input=False,predictor_updates=1000,predictor_lr=2e-4,
        sample=data.summary,full80_gate='heldoutNLL decreases>0.03 and accuracy increases>1pp and changed-state accuracy does not fall',
        max_actions=32,validation_used_for_optimization=False,finalfixed80_used=False)
    if rank==0:atomic(OUT/'manifest.json',manifest);atomic(OUT/'dataset.json',data.summary)

    @torch.no_grad()
    def score(update):
        predictor.eval();totals=torch.zeros(10,device=device,dtype=torch.float64)
        for ids in data.holdout[rank::2].split(1024):
            inputs=data.inputs(ids);lp=predictor(**inputs);target=data.columns['target'][ids]
            changed=data.columns['changed'][ids];hit=lp.argmax(-1)==target
            base=inputs['base_logp'];basehit=base.argmax(-1)==target
            nll=F.nll_loss(lp,target,reduction='sum');bnll=F.nll_loss(base,target,reduction='sum')
            # Evaluation-only ablation of this SAME predictor, no second model.
            current=predictor(**data.inputs(ids,history=False))
            vals=[len(ids),nll,hit.sum(),changed.sum(),(hit&changed).sum(),bnll,basehit.sum(),
                  (basehit&changed).sum(),F.nll_loss(current,target,reduction='sum'),(current.argmax(-1)==target).sum()]
            totals+=torch.stack([torch.as_tensor(v,device=device,dtype=torch.float64) for v in vals])
        dist.all_reduce(totals);n,loss,h,c,ch,bl,bh,bch,nohist,nh=totals.cpu().tolist()
        row=dict(update=update,time=time.time(),rows=int(n),nll=loss/n,accuracy=h/n,changed_accuracy=ch/max(1,c),
            source_nll=bl/n,source_accuracy=bh/n,source_changed_accuracy=bch/max(1,c),
            same_predictor_without_history_nll=nohist/n,same_predictor_without_history_accuracy=nh/n)
        if rank==0:log('heldout.jsonl',row)
        predictor.train();return row

    predictor.train();batch,capacity=choose(predictor,data,rng)
    if rank==0:atomic(OUT/'capacity.json',capacity)
    wrapper=DDP(predictor,device_ids=[local],broadcast_buffers=False)
    baseline=score(0)
    if rank==0:atomic(OUT/'baseline_holdout.json',baseline)
    best=float('inf');chosen=None;began=time.time()
    for update in range(1,1001):
        tick=time.time();ids=data.sample(rng,batch);inputs=data.inputs(ids);target=data.columns['target'][ids]
        optimizer.zero_grad(set_to_none=True);lp=wrapper(**inputs)
        nll=F.nll_loss(lp,target)
        kl=F.kl_div(lp,inputs['base_logp'].softmax(-1),reduction='batchmean')
        loss=nll+.2*kl;loss.backward();norm=torch.nn.utils.clip_grad_norm_(predictor.parameters(),5)
        assert torch.isfinite(loss) and torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
        row=dict(update=update,time=time.time(),seconds=time.time()-tick,loss=float(loss),nll=float(nll),
            anchor_kl=float(kl),gradient_norm=float(norm),batch_per_gpu=batch,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        log(f'rank{rank}.jsonl',row);atomic(OUT/f'status_rank{rank}.json',dict(state='training',pid=os.getpid(),**row))
        if False: # Full-data capacity was measured before any update.
            peak=torch.tensor(row['peak_gib'],device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
            if peak<8:batch=8192
            if rank==0:atomic(OUT/'capacity.json',dict(measured_training_peak_gib=float(peak),batch_per_gpu=batch))
        if update%50==0:
            measured=score(update)
            qualifies=(measured['nll']<baseline['nll']-.03 and measured['accuracy']>baseline['accuracy']+.01 and
                       measured['changed_accuracy']>=baseline['changed_accuracy'])
            newbest=qualifies and measured['nll']<best
            if newbest:chosen=update;best=measured['nll']
            if rank==0:
                log('selection.jsonl',dict(**measured,qualifies=qualifies,chosen=chosen,elapsed=time.time()-began))
                destination=OUT/f'predictor_{update:06d}.pt';tmp=destination.with_suffix('.tmp')
                torch.save(dict(predictor=predictor.state_dict(),optimizer=optimizer.state_dict(),manifest=manifest,
                    source_commit=commit,update=update,best=best,chosen=chosen),tmp);tmp.replace(destination)
                record=dict(filename=destination.name,update=update,sha256=sha(destination),time=time.time())
                atomic(destination.with_suffix('.COMMITTED.json'),record);atomic(OUT/'latest.COMMITTED.json',record)
                if newbest:atomic(OUT/'best.COMMITTED.json',record)
                keep={destination.name,f'predictor_{chosen:06d}.pt' if chosen else ''}
                for old in OUT.glob('predictor_*.pt'):
                    if old.name not in keep and old.with_suffix('.COMMITTED.json').exists():
                        old.unlink();old.with_suffix('.COMMITTED.json').unlink()
            dist.barrier()
    summary=None;selected=None
    if chosen is not None:
        selected=json.loads((OUT/'best.COMMITTED.json').read_text());p=OUT/selected['filename'];assert sha(p)==selected['sha256']
        predictor.load_state_dict(torch.load(p,map_location=device,weights_only=False)['predictor']);predictor.eval()
        del wrapper,optimizer,data,inputs,loss,lp,kl,nll;gc.collect();torch.cuda.empty_cache()
        atomic(OUT/f'status_rank{rank}.json',dict(state='evaluating_new80',pid=os.getpid(),update=chosen,time=time.time()))
        rows=evaluate(model,manifest,OUT/'development_best',rank,2,workers=40,history_predictor=predictor)
        all_rows=[None,None];dist.all_gather_object(all_rows,rows)
        if rank==0:atomic(OUT/'development_best/summary.json',summarize([r for rows in all_rows for r in rows]))
        dist.barrier();summary=json.loads((OUT/'development_best/summary.json').read_text())
    atomic(OUT/f'status_rank{rank}.json',dict(state='complete',pid=os.getpid(),time=time.time(),updates=1000))
    if rank==0:atomic(OUT/'result.json',dict(state='complete',updates=1000,time=time.time(),chosen=selected,
        development=summary,source_development_successes=44,source_episodes=80,goal_complete=False,finalfixed80_used=False))
    dist.destroy_process_group()

if __name__=='__main__':main()
