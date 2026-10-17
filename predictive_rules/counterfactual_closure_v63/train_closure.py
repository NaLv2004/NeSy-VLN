"""Two-GPU fixed-abstraction rule fitting and one unchanged full80 trial."""
import argparse,json,os,random,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import FactoredWorld,Config
from closure_objective import ClosureObjective,configure,TRAINABLE
from closure_data import ClosureData
from evaluate import atomic,evaluate,summarize
from data import sha,ROOT
from accelerated_transition import install

OUT=ROOT/'run_counterfactual_closure_v63'
DATA=ROOT/'counterfactual_closure_v63'

def write_log(name,row):
    with (OUT/name).open('a') as f:f.write(json.dumps(row)+'\n')


def aggregate(items,device):
    total={}
    counts={'first_o_hits','first_o_count','changed_hits','changed_count','first_m_hits',
            'conditional_o_hits','conditional_m_hits','open_o_hits','open_hard_m_hits','second_count','positions'}
    for row in items:
        n=float(row['positions'])
        for k,v in row.items(): total[k]=total.get(k,0.)+float(v)*(1. if k in counts else n)
    keys=sorted(total)
    sums=torch.tensor([total[k] for k in keys],device=device,dtype=torch.float64)
    dist.all_reduce(sums);total=dict(zip(keys,sums.cpu().tolist()))
    result={k: v/max(1,total['positions']) for k,v in total.items() if k not in counts}
    result.update(positions=int(total['positions']),
        first_o_accuracy=total['first_o_hits']/max(1,total['first_o_count']),
        changed_accuracy=total['changed_hits']/max(1,total['changed_count']),
        first_m_accuracy=total['first_m_hits']/max(1,total['first_o_count']))
    for k in ('conditional_o','conditional_m','open_o','open_hard_m'):
        result[k+'_accuracy']=total[k+'_hits']/max(1,total['second_count'])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--batch',type=int,default=256);parser.add_argument('--resume');args=parser.parse_args()
    if (OUT/'efficiency.json').exists():
        args.batch=int(json.loads((OUT/'efficiency.json').read_text())['batch_per_gpu'])
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6,7' and int(os.environ['WORLD_SIZE'])==2
    torch.cuda.set_device(local);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dist.init_process_group('nccl');torch.manual_seed(631017)
    rng=random.Random(631017+rank);device=torch.device('cuda',local)
    OUT.mkdir(exist_ok=True);data=ClosureData(DATA)
    source=torch.load(DATA/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**source['model_config'])).to(device)
    model.load_state_dict(source['model'],strict=True)
    install(model)
    # This compact categorical dataset fits easily on each GPU. Transfer once
    # instead of copying nine small tensors in every optimizer update.
    data.tensors={part:{k:v.to(device) for k,v in rows.items()} for part,rows in data.tensors.items()}
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith(TRAINABLE)}
    params=configure(model);objective=ClosureObjective(model)
    ddp=DDP(objective,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    manifest=dict(source['manifest']);manifest.update(architecture='counterfactual_closure_v63',
        source_checkpoint=data.commit,dataset_manifest_sha256=data.fingerprint,fit_episodes=data.splits['fit'],
        transition_holdout_episodes=data.splits['holdout'],trainable_prefixes=TRAINABLE,
        abstraction_and_goal_frozen=True,updates=args.updates,batch_per_gpu=args.batch,
        second_pairs_per_position=8,learning_rate=1e-4,gradient_clip=5.,
        labels='Actual measured observations encoded by frozen source abstraction; copied offline memory grounder for targets only',
        objective='1step O/M CE +0.5*(2step observed-O/predicted-M CE for O/M + exact64-intermediate-O marginal likelihood for O/M)',
        discrete_forward=True,neural_planning=False,per_action_memory_grounding=False,
        finalfixed80_used=False,full_dev80_unchanged=True,
        source_reference='https://planetrl.github.io/',
        adaptation='Multi-step latent supervision only; no PlaNet reward model, continuous state, filtering, or CEM planner',
        checkpoint_selection='Minimum total transition holdout NLL at preselected50-update intervals; original full dev80 run once after fitting',
        efficiency='Pruned unused legacy O decoder; GPU resident categorical data; batched random pair selection',
        transition_holdout_interval=50,transition_holdout_batch=128)
    del source
    first=1;best=float('inf');history=[]
    if args.resume:
        p=Path(args.resume);c=json.loads(p.with_suffix('.COMMITTED.json').read_text());assert sha(p)==c['sha256']
        saved=torch.load(p,map_location=device,weights_only=False)
        assert saved['manifest']['dataset_manifest_sha256']==data.fingerprint
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        first=saved['update']+1;best=saved['best_score'];history=saved['history']
        rs=saved['rank_rng_states'][rank];rng.setstate(rs['python_rng'])
        torch.set_rng_state(rs['torch_rng'].cpu());torch.cuda.set_rng_state(rs['cuda_rng'].cpu());del saved
    if rank==0:atomic(OUT/'manifest.json',manifest)

    @torch.no_grad()
    def heldout(update):
        model.eval();items=[]
        indices=torch.arange(len(data.tensors['holdout']['memory']))[rank::2]
        for idx in indices.split(128):
            _,metrics=objective(data.batch('holdout',idx,device));items.append(metrics)
        result=aggregate(items,device);row=dict(update=update,time=time.time(),**result)
        if rank==0:write_log('transitions.jsonl',row)
        model.train();return row

    def save(update):
        states=[None,None]
        dist.all_gather_object(states,dict(python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        if rank==0:
            name=f'checkpoint_{update:06d}.pt';p=OUT/name;tmp=p.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),model_config=model.export_config(),optimizer=optimizer.state_dict(),
                update=update,manifest=manifest,rank_rng_states=states,best_score=best,history=history),tmp)
            tmp.replace(p);c=dict(filename=name,update=update,sha256=sha(p),time=time.time())
            atomic(p.with_suffix('.COMMITTED.json'),c);atomic(OUT/'latest.COMMITTED.json',c)
            return c

    if first==1:history.append(heldout(0))
    began=time.time()
    for update in range(first,args.updates+1):
        t=time.time();model.train()
        d=data.sample(rng,args.batch,device)
        pairs=torch.rand(args.batch,32,device=device).argsort(dim=1)[:,:8]+2
        loss,metrics=ddp(d,pairs);optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,5.)
        assert torch.isfinite(loss) and torch.isfinite(norm);optimizer.step()
        row=dict(update=update,time=time.time(),seconds=time.time()-t,gradient_norm=float(norm),
                 gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                 **{k:float(v) for k,v in metrics.items()})
        write_log(f'rank{rank}.jsonl',row)
        atomic(OUT/f'status_rank{rank}.json',dict(state='training',pid=os.getpid(),update=update,time=time.time()))
        if update%50==0 or update==args.updates:
            held=heldout(update);history.append(held);improved=held['loss']<best
            best=min(best,held['loss'])
            if rank==0:
                for k,v in frozen.items():assert torch.equal(v,model.state_dict()[k].detach().cpu()),k
            commit=save(update)
            if rank==0:
                if improved:atomic(OUT/'best.COMMITTED.json',commit)
                keep={commit['filename'],json.loads((OUT/'best.COMMITTED.json').read_text())['filename']}
                for old in OUT.glob('checkpoint_*.pt'):
                    if old.name not in keep:
                        assert old.parent==OUT and old.with_suffix('.COMMITTED.json').exists()
                        old.unlink();old.with_suffix('.COMMITTED.json').unlink()
                atomic(OUT/'frozen_contract.json',dict(update=update,frozen_tensors=len(frozen),identical_to_source=True,
                    no_visual_or_memory_grounder_or_goal_changes=True,trainable_prefixes=TRAINABLE))
                print(json.dumps(dict(update=update,heldout=held,elapsed=time.time()-began)),flush=True)
            dist.barrier()

    dist.barrier();commit=json.loads((OUT/'best.COMMITTED.json').read_text());p=OUT/commit['filename']
    assert sha(p)==commit['sha256'];saved=torch.load(p,map_location=device,weights_only=False)
    model.load_state_dict(saved['model']);del saved
    model.detach_memory_carrier=True;model.eval()
    atomic(OUT/f'status_rank{rank}.json',dict(state='evaluating_full80',pid=os.getpid(),time=time.time(),update=commit['update']))
    folder=OUT/'dev_best_full80';summary=folder/'summary.json'
    # A completed trial is never repeated after a supervisor recovery.
    if not summary.exists():
        rows=evaluate(model,manifest,folder,rank=rank,world_size=2,workers=8)
        collected=[None,None];dist.all_gather_object(collected,rows)
        if rank==0:
            merged=[r for rs in collected for r in rs];assert len(merged)==80
            atomic(summary,summarize(merged))
    dist.barrier()
    atomic(OUT/f'status_rank{rank}.json',dict(state='complete',pid=os.getpid(),time=time.time(),updates=args.updates))
    if rank==0:atomic(OUT/'result.json',dict(state='complete',time=time.time(),updates=args.updates,
        evaluated_checkpoint=commit,transitions=history,development=json.loads(summary.read_text()),
        finalfixed80_used=False,completion_proven=False))
    dist.destroy_process_group()

if __name__=='__main__':main()
