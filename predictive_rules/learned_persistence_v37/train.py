"""Two-GPU joint representation/rule training with immutable split evaluation."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import time
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from model import FactoredWorld,Config,hard
from data import Corpus,ROOT,sha
from evaluate import atomic,evaluate,summarize

def append(path,value):
    with open(path,'a') as f:f.write(json.dumps(value)+'\n')

def save(out,model,optimizer,update,manifest,rng,rank_states=None):
    filename=f'checkpoint_{update:06d}.pt';path=out/filename;temp=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),model_config=model.export_config(),optimizer=optimizer.state_dict(),
        update=update,manifest=manifest,python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),rank_rng_states=rank_states),temp)
    temp.replace(path);commit=dict(filename=filename,update=update,sha256=sha(path),time=time.time())
    atomic(path.with_suffix('.COMMITTED.json'),commit);atomic(out/'latest.COMMITTED.json',commit)
    # Keep latest two plus every best checkpoint explicitly pointed to.
    keep=set(sorted(out.glob('checkpoint_*.pt'))[-2:])
    if (out/'best.COMMITTED.json').exists():keep.add(out/json.loads((out/'best.COMMITTED.json').read_text())['filename'])
    for old in out.glob('checkpoint_*.pt'):
        if old not in keep:
            assert old.parent==out and old.with_suffix('.COMMITTED.json').exists()
            old.unlink();old.with_suffix('.COMMITTED.json').unlink()
    return commit

@torch.no_grad()
def initialize(model,corpus,rng):
    data=corpus.sample(rng,128,2,'cuda')
    x=data['features'];mask=torch.arange(x.shape[1],device=x.device)[None]<=data['end'][:,None]
    z=F.normalize(model.visual(x.float().flatten(-2)),dim=-1)[mask]
    def kmeans(values,count):
        centers=values[torch.randperm(len(values),device=values.device)[:count]].clone()
        for _ in range(20):
            assignment=(values@centers.T).argmax(-1)
            sums=torch.zeros_like(centers).index_add_(0,assignment,values)
            counts=torch.bincount(assignment,minlength=count)
            centers=torch.where(counts[:,None]>0,F.normalize(sums,dim=-1),centers)
        return centers
    model.observation_definitions.copy_(kmeans(z,model.cfg.observations))
    observations=hard(model.observe(x));_,states,_=model.memory_sequence(observations,data['previous'])
    model.memory_definitions.copy_(kmeans(F.normalize(states,dim=-1)[mask],model.cfg.memories))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--updates',type=int,default=10000)
    parser.add_argument('--batch-size',type=int,default=128);parser.add_argument('--branch-batch',type=int,default=64)
    parser.add_argument('--eval-every',type=int,default=500);parser.add_argument('--resume');parser.add_argument('--initialize');args=parser.parse_args()
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6,7'
    torch.cuda.set_device(local);torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if world>1:dist.init_process_group('nccl')
    torch.manual_seed(371017);rng=random.Random(371017+rank)
    out=Path(args.out).resolve();assert out.is_relative_to(ROOT);out.mkdir(exist_ok=True,parents=True)
    corpus=Corpus(rank,world);model=FactoredWorld().cuda()
    if not args.initialize and not args.resume:initialize(model,corpus,rng)
    source_sha=None
    if args.initialize and not args.resume:
        source_sha=sha(args.initialize)
        source_commit=json.loads(Path(args.initialize).with_suffix('.COMMITTED.json').read_text())
        assert source_sha==source_commit['sha256']
        previous=torch.load(args.initialize,map_location='cuda',weights_only=False)
        missing,unexpected=model.load_state_dict(previous['model'],strict=False)
        assert not unexpected and all(n.startswith(('persistence.','action_persistence.')) for n in missing)
        del previous
    if world>1:
        for parameter in model.parameters():dist.broadcast(parameter,src=0)
    representation_prefixes=('visual.','observation_definitions','memory.','memory_definitions')
    optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if n.startswith(representation_prefixes)],lr=2e-5),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith(representation_prefixes)],lr=1e-4)],weight_decay=1e-4)
    manifest=dict(architecture='learned_persistence_world_v37',data=corpus.manifest['data'],frozen_vision=corpus.manifest['frozen_vision'],
        corpus=corpus.provenance,model_config=model.export_config(),training_args=vars(args),created=time.time(),
        parameters=sum(p.numel() for p in model.parameters()),
        observation_state='Frame-only categorical64, learned global definitions, no handcrafted semantics',
        memory_state='Categorical32, initialized from discrete observation/action history only at rule regeneration',
        rule='Two complete categorical tables with learned preserve-or-replace probabilities for all action IDs; hard argmax export',
        dynamics_task_invariant=True,task_conditioning='Only the goal generator sees task identity; world dynamics and learned state definitions cannot depend on the requested goal',
        identity_operator_manually_assigned=False,persistence_targets='Equality of independently encoded recorded before/after observations and memories',
        initialize=args.initialize,initialize_sha256=source_sha,optimizer_initialization='Fresh AdamW; existing learned model weights retained except new persistence gates',
        learning_rates=dict(representation=2e-5,world_goal_and_auxiliary=1e-4),
        planner='Exact backward derivation over joint integer state, final STOP at unchanged generated goal, no neural action fallback',
        rule_supervision='Recorded successors for all17 actions plus autonomous10-step symbolic recurrence',
        no_map=True,no_collision_state_or_value_table=True,no_validation_gradient=True,
        inverse_training_only=True,inverse_inputs='Two hard observation IDs plus separation; memory and action history forbidden',
        renewal='Learned discrepancy probability with minimum five executed actions',
        references=['https://github.com/midi-lab/acdf','https://arxiv.org/abs/2107.00110'],fixed_development_episodes=80,max_actions=32)
    first=1
    if args.resume:
        saved=torch.load(args.resume,map_location='cuda',weights_only=False)
        weight=saved['model']['inverse.0.weight'];selected=None;c=model.cfg
        if weight.shape[1]!=model.inverse[0].weight.shape[1]:
            selected=list(range(c.memories,c.memories+c.observations))+list(range(2*c.memories+c.observations,2*(c.memories+c.observations)+10))
            saved['model']['inverse.0.weight']=weight[:,selected]
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);first=saved['update']+1
        if selected is not None:
            state=optimizer.state[model.inverse[0].weight]
            for key in ('exp_avg','exp_avg_sq'):
                if key in state:state[key]=state[key][:,selected].contiguous()
            manifest['inverse_leakage_fix_at_update']=saved['update']
        states=saved.get('rank_rng_states')
        state=states[rank] if states else saved
        rng.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        if not states and rank:
            rng.seed(361017+rank+100000*saved['update']);torch.manual_seed(361017+rank+100000*saved['update'])
    if rank==0:atomic(out/'manifest.json',manifest)
    wrapper=DDP(model,device_ids=[local],find_unused_parameters=False) if world>1 else model
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    began=time.time();best=-1
    if (out/'development.jsonl').exists():
        best=max([json.loads(line)['successes'] for line in (out/'development.jsonl').read_text().splitlines() if line.strip()]+[-1])
    for update in range(first,args.updates+1):
        tick=time.time();model.train();data=corpus.sample(rng,args.batch_size,args.branch_batch,'cuda')
        optimizer.zero_grad(set_to_none=True)
        loss,metrics=wrapper(data)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; last COMMITTED checkpoint preserved')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient; checkpoint preserved')
        optimizer.step();torch.cuda.synchronize()
        row={k:float(v) for k,v in metrics.items()};row.update(update=update,time=time.time(),seconds=time.time()-tick,gradient_norm=float(norm),rank=rank,
            memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,state='training',pid=os.getpid())
        append(out/f'rank{rank}.jsonl',row);atomic(out/f'rank{rank}.status.json',row)
        if rank==0:print(json.dumps(row),flush=True)
        stopping=torch.tensor([int(stop[0])],device='cuda')
        if world>1:dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
        commit=None
        if update%100==0 or update==args.updates or bool(stopping):
            state=dict(python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            states=[state]
            if world>1:
                states=[None]*world;dist.all_gather_object(states,state)
            if rank==0:commit=save(out,model,optimizer,update,manifest,rng,states)
        if update%args.eval_every==0:
            if world>1:dist.barrier()
            atomic(out/f'rank{rank}.status.json',dict(row,state='evaluating',time=time.time()))
            # Both cards evaluate disjoint halves of precisely the same dev80.
            rows=evaluate(model,manifest,out/f'dev_{update:06d}',rank,world)
            if world>1:
                gathered=[None]*world;dist.all_gather_object(gathered,rows);rows=sum(gathered,[])
            if rank==0:
                assert len(rows)==80 and len({r['episode_id'] for r in rows})==80
                result=summarize(rows);result.update(update=update,time=time.time())
                atomic(out/f'dev_{update:06d}/summary.json',result);append(out/'development.jsonl',result)
                if result['successes']>best:
                    best=result['successes'];atomic(out/'best.COMMITTED.json',commit or json.loads((out/'latest.COMMITTED.json').read_text()))
                print('DEVELOPMENT '+json.dumps(result),flush=True)
            if world>1:dist.barrier()
        if bool(stopping):break
    if rank==0:atomic(out/'status.json',dict(state='paused' if stop[0] else 'finished',update=update,time=time.time(),elapsed=time.time()-began))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
