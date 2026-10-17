"""Restricted all-symbol visual evidence adaptation; no neural action head."""
import json,os,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import FactoredWorld,Config,hard
from data import ROOT,sha
from feedback_objective import FeedbackData
from evaluate import atomic,evaluate,summarize

DATA=ROOT/'goal_discrimination_v64';OUT=ROOT/'run_linear_evidence_v65'

def log(name,row):
    with (OUT/name).open('a') as f:f.write(json.dumps(row)+'\n')

@torch.no_grad()
def prepare(model,data,rank):
    contexts=[];base_o=[]
    for start in range(0,len(data.features),32):
        x=data.features[start:start+32];previous=data.previous[start:start+32];task=data.tasks[start:start+32]
        global_o=hard(model.observe(x))
        contexts.append(model.context_sequence(x,global_o,previous,task))
    contexts=torch.cat(contexts);result={}
    for part,all_rows in data.parts.items():
        rows=all_rows[rank::2]
        index=torch.tensor([r[0] for r in rows],device=contexts.device)
        current=torch.tensor([r[1] for r in rows],device=contexts.device)
        boundary=torch.tensor([r[2] for r in rows],device=contexts.device)
        context=contexts[index,boundary];x=data.features[index,current].float()
        logits=model.observe(x,context)
        definitions=F.normalize(model.observation_definitions[None]+model.observation_definition_delta(context),dim=-1)
        _,go=model.goal_logits(context,data.tasks[index])
        y=torch.tensor([r[3] for r in rows],device=x.device)
        result[part]=dict(x=x.flatten(-2),context=context,definitions=definitions,base=logits,
            goal=go.argmax(-1),y=y,positive=torch.where(y)[0],negative=torch.where(~y)[0])
    return result

def calculate(model,d,index=None):
    x=d['x'] if index is None else d['x'][index]
    definitions=d['definitions'] if index is None else d['definitions'][index]
    base=d['base'] if index is None else d['base'][index]
    delta=model((x-model.module_mean)/model.module_scale)
    return base+torch.einsum('bw,bow->bo',delta,definitions)

def main():
    rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6,7' and int(os.environ['WORLD_SIZE'])==2
    torch.cuda.set_device(device);torch.set_num_threads(3);torch.manual_seed(651017)
    dist.init_process_group('nccl');OUT.mkdir(exist_ok=True)
    assert not (OUT/'result.json').exists()
    commit=json.loads((DATA/'source.COMMITTED.json').read_text());assert sha(DATA/'source.pt')==commit['sha256']
    source=torch.load(DATA/'source.pt',map_location='cpu',weights_only=False)
    config=dict(source['model_config'],linear_visual_evidence=True)
    model=FactoredWorld(Config(**config)).to(device)
    missing,unexpected=model.load_state_dict(source['model'],strict=False)
    assert not unexpected and set(missing)=={'linear_visual_evidence.weight','evidence_mean','evidence_scale'}
    model.eval().requires_grad_(False)
    data=FeedbackData(DATA,source['manifest']['transition_holdout_episodes'],device)
    parts=prepare(model,data,rank);d=parts['fit'];del data
    # Normalization uses fitting frames only, all-reduced across both shards.
    sums=torch.stack([d['x'].sum(0),d['x'].square().sum(0)])
    count=torch.tensor(float(len(d['x'])),device=device);dist.all_reduce(sums);dist.all_reduce(count)
    mean=sums[0]/count;scale=(sums[1]/count-mean.square()).clamp_min(.0004).sqrt()
    model.evidence_mean.copy_(mean);model.evidence_scale.copy_(scale)
    model.linear_visual_evidence.requires_grad_(True)
    wrapper=DDP(model.linear_visual_evidence,device_ids=[device.index],broadcast_buffers=False)
    wrapper.module_mean=mean;wrapper.module_scale=scale
    optimizer=torch.optim.AdamW(wrapper.parameters(),lr=2e-4,weight_decay=.1)
    rng=torch.Generator(device=device).manual_seed(651017+rank)
    manifest=dict(source['manifest']);del source
    manifest.update(architecture='linear_evidence_v65',source_commit=commit,
        trained_parameters=['linear_visual_evidence.weight'],all_original_parameters_frozen=True,
        all_code_linear_evidence=True,inputs_to_extra_encoder='current visual patches and generated anonymous definitions only',
        no_fixed_stop_code=True,no_runtime_goal_feedback=True,no_new_runtime_action_head=True,
        updates=400,batch_positions_per_gpu=4096,learning_rate=2e-4,weight_decay=.1,
        feedback_weight=1.,anchor_kl_weight=1.,residual_square_weight=.001,
        train_tasks=480,feedback_holdout_tasks=120,finalfixed80_used=False,original_full_dev80=True,
        runtime_planner='Unchanged discrete probability-table derivation',
        selection='Lowest heldout balanced NLL with TPR>=baseline+0.1, FPR<=0.02, negative non-goal original code agreement>=0.90')
    if rank==0:atomic(OUT/'manifest.json',manifest)

    @torch.no_grad()
    def score(update):
        d=parts['holdout'];totals=torch.zeros(8,device=device,dtype=torch.float64)
        for start in range(0,len(d['x']),1024):
            ids=torch.arange(start,min(start+1024,len(d['x'])),device=device)
            logits=calculate(wrapper,d,ids);goal=d['goal'][ids];y=d['y'][ids]
            logp=logits.log_softmax(-1).gather(-1,goal[:,None]).squeeze(-1)
            p=logp.exp().clamp_max(1-1e-6);pred=logits.argmax(-1)==goal
            anchor=(~y)&(d['base'][ids].argmax(-1)!=goal)
            correct=(logits.argmax(-1)==d['base'][ids].argmax(-1))&anchor
            totals+=torch.stack([y.sum(),(~y).sum(),(pred&y).sum(),(pred&~y).sum(),
                (-logp*y).sum(),(-torch.log1p(-p)*(~y)).sum(),anchor.sum(),correct.sum()]).double()
        dist.all_reduce(totals);p,n,tp,fp,pl,nl,anc,correct=totals.cpu().tolist()
        row=dict(update=update,time=time.time(),positive=int(p),negative=int(n),tp=int(tp),fp=int(fp),
            tpr=tp/p,fpr=fp/n,balanced_nll=.5*(pl/p+nl/n),negative_code_agreement=correct/max(1,anc))
        if rank==0:log('heldout.jsonl',row)
        return row

    baseline=score(0);best=None;best_nll=float('inf');began=time.time()
    for update in range(1,401):
        tick=time.time();positive=d['positive'];negative=d['negative']
        ids=torch.cat([positive[torch.randint(len(positive),(2048,),generator=rng,device=device)],
            negative[torch.randint(len(negative),(2048,),generator=rng,device=device)]])
        optimizer.zero_grad(set_to_none=True);logits=calculate(wrapper,d,ids)
        goal=d['goal'][ids];y=d['y'][ids];logp=logits.log_softmax(-1).gather(-1,goal[:,None]).squeeze(-1)
        p=logp.exp().clamp_max(1-1e-6);feedback=torch.where(y,-logp,-torch.log1p(-p)).mean()
        anchor=(~y)&(d['base'][ids].argmax(-1)!=goal)
        kl=F.kl_div(logits.log_softmax(-1),d['base'][ids].softmax(-1),reduction='none').sum(-1)
        anchor_loss=(kl*anchor).sum()/anchor.sum().clamp_min(1)
        residual=(logits-d['base'][ids]).square().mean()
        loss=feedback+anchor_loss+.001*residual
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(wrapper.parameters(),5.)
        assert torch.isfinite(norm) and torch.isfinite(loss)
        optimizer.step();torch.cuda.synchronize()
        row=dict(update=update,time=time.time(),seconds=time.time()-tick,loss=float(loss),feedback_loss=float(feedback),
            anchor_kl=float(anchor_loss),residual_square=float(residual),gradient_norm=float(norm),
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        log(f'rank{rank}.jsonl',row)
        atomic(OUT/f'status_rank{rank}.json',dict(state='training',pid=os.getpid(),**row))
        if update%50==0:
            measured=score(update)
            qualifies=measured['tpr']>=baseline['tpr']+.1 and measured['fpr']<=.02 and measured['negative_code_agreement']>=.9
            if qualifies and measured['balanced_nll']<best_nll:
                best=measured;best_nll=measured['balanced_nll']
                if rank==0:
                    path=OUT/'best.pt';temp=path.with_suffix('.tmp')
                    torch.save(dict(model=model.state_dict(),model_config=model.export_config(),manifest=manifest,update=update,
                        optimizer=optimizer.state_dict()),temp);temp.replace(path)
                    atomic(OUT/'best.COMMITTED.json',dict(filename=path.name,update=update,sha256=sha(path),time=time.time()))
            if rank==0:print(json.dumps(dict(update=update,heldout=measured,qualifies=qualifies,elapsed=time.time()-began)),flush=True)
            dist.barrier()
    if rank==0:
        path=OUT/'last.pt';temp=path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),model_config=model.export_config(),manifest=manifest,update=400,
            optimizer=optimizer.state_dict()),temp);temp.replace(path)
        atomic(OUT/'last.COMMITTED.json',dict(filename=path.name,update=400,sha256=sha(path),time=time.time()))
    dist.barrier()
    development=None;chosen=None
    if best is not None:
        chosen=json.loads((OUT/'best.COMMITTED.json').read_text());path=OUT/chosen['filename'];assert sha(path)==chosen['sha256']
        model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model'])
        atomic(OUT/f'status_rank{rank}.json',dict(state='evaluating_full80',pid=os.getpid(),update=best['update'],time=time.time()))
        rows=evaluate(model,manifest,OUT/'dev_best_full80',rank,2,workers=12)
        all_rows=[None,None];dist.all_gather_object(all_rows,rows)
        if rank==0:
            development=summarize([r for group in all_rows for r in group]);atomic(OUT/'dev_best_full80/summary.json',development)
    dist.barrier();atomic(OUT/f'status_rank{rank}.json',dict(state='complete',pid=os.getpid(),updates=400,time=time.time()))
    if rank==0:atomic(OUT/'result.json',dict(state='complete',time=time.time(),baseline=baseline,best=best,
        development=development,evaluated_checkpoint=chosen,full80_run=best is not None,finalfixed80_used=False,goal_complete=False))
    dist.destroy_process_group()

if __name__=='__main__':main()
