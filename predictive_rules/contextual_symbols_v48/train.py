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
from storage import retain_evaluation_tables

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
    logits,evidence,_=model.observation_encoding(x)
    observations=hard(logits);_,states,_=model.memory_sequence(observations,data['previous'],evidence)
    model.memory_definitions.copy_(kmeans(F.normalize(states,dim=-1)[mask],model.cfg.memories))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--updates',type=int,default=10000)
    parser.add_argument('--batch-size',type=int,default=128);parser.add_argument('--branch-batch',type=int,default=64)
    parser.add_argument('--branch-root');parser.add_argument('--eval-every',type=int,default=500);parser.add_argument('--resume');parser.add_argument('--initialize');args=parser.parse_args()
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6,7'
    torch.cuda.set_device(local);torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if world>1:dist.init_process_group('nccl')
    torch.manual_seed(391017);rng=random.Random(391017+rank)
    out=Path(args.out).resolve();assert out.is_relative_to(ROOT);out.mkdir(exist_ok=True,parents=True)
    corpus=Corpus(rank,world,args.branch_root);model=FactoredWorld().cuda()
    if not args.initialize and not args.resume:initialize(model,corpus,rng)
    source_sha=None
    if args.initialize and not args.resume:
        source_sha=sha(args.initialize)
        source_commit=json.loads(Path(args.initialize).with_suffix('.COMMITTED.json').read_text())
        assert source_sha==source_commit['sha256']
        previous=torch.load(args.initialize,map_location='cuda',weights_only=False)
        assert previous['manifest']['data']==corpus.manifest['data'], 'Training/validation split changed'
        model.load_v43(previous['model'])
        del previous
    if world>1:
        for parameter in model.parameters():dist.broadcast(parameter,src=0)
    representation_prefixes=('visual.','observation_definitions','memory.','memory_definitions')
    optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if n.startswith(representation_prefixes)],lr=2e-5),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith(representation_prefixes)],lr=1e-4)],weight_decay=1e-4)
    manifest=dict(architecture='contextual_symbols_v48',data=corpus.manifest['data'],frozen_vision=corpus.manifest['frozen_vision'],
        corpus=corpus.provenance,model_config=model.export_config(),training_args=vars(args),created=time.time(),
        parameters=sum(p.numel() for p in model.parameters()),
        observation_state='Categorical64 with full128-dimensional definitions generated from boundary history; fixed until rule renewal, no handcrafted semantics',
        memory_state='Categorical32 predicted by independent rule network from current discrete premises; real visual history grounds memory only when renewing rules',
        rule='Export deterministic next-memory IDs and a probability distribution over all 64 possible next observation IDs for every current joint state and action',
        joint_successor_consistency='Next memory is determined from current available discrete premises, independently of the later observed outcome',
        dynamics_task_invariant=True,task_conditioning='Only the goal generator sees task identity; world dynamics and learned state definitions cannot depend on the requested goal',
        identity_operator_manually_assigned=False,persistence_targets='Equality of independently encoded recorded before/after observations and memories',
        initialize=args.initialize,initialize_sha256=source_sha,optimizer_initialization='v43 best weights initialize existing modules; new full definition vectors zero-initialized; fresh AdamW',
        memory_supervision='Actual arriving visual evidence and history encode target IDs; separate current-only predictor is supervised by those IDs',
        memory_grounding_auxiliary='Reconstruct recorded raw patch features from joint observation/memory codes; no semantic/collision state labels',
        memory_grounding_runtime='Only during explicit rule renewal, never after an ordinary action',
        operator_classes='Shared preserve-or-change class learned from before/after effects of every opaque action ID on fitting training episodes',
        operator_class_gradient='Raw-outcome classification only; world and goal losses cannot alter operator labels',
        symbolic_backward='Detach copied categorical observation evidence and previous memory carrier; identical forward hard states and rule execution',
        learning_rates=dict(representation=2e-5,world_goal_and_auxiliary=1e-4),
        planner='Finite-horizon probability derivation over integer states and all learned outcomes; unchanged exact generated goal; no neural action fallback or learned value/reward table',
        rule_supervision='Recorded successors for all17 actions plus autonomous10-step symbolic recurrence',
        no_map=True,no_collision_state_or_value_table=True,no_validation_gradient=True,
        observation_quantization_loss='Spherical VQ commitment0.25 plus codebook fit; combined visual/memory term weighted0.25',world_to_observation_encoder_every=5,inverse_training_only=True,inverse_inputs='Two hard observation IDs plus separation; memory and action history forbidden',
        renewal='Learned discrepancy probability with minimum five executed actions',
        references=['https://github.com/midi-lab/acdf','https://arxiv.org/abs/2107.00110'],fixed_development_episodes=80,max_actions=32)
    first=1
    if args.resume:
        resume_commit=json.loads(Path(args.resume).with_suffix('.COMMITTED.json').read_text())
        assert sha(args.resume)==resume_commit['sha256']
        saved=torch.load(args.resume,map_location='cuda',weights_only=False)
        manifest['resume_commit']=resume_commit
        manifest['resume_path']=args.resume
        manifest['resume_architecture']=saved['manifest']['architecture']
        assert manifest['resume_architecture'] in ('contextual_symbols_v48',)
        assert saved['manifest']['data']==manifest['data'], 'Training/validation split changed'
        assert len(saved['rank_rng_states'])==world, 'Preserve per-rank random state'
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
    manifest.update(observation_definitions_generated_in_full=True,
        observation_definitions_frozen_between_renewals=True,
        predicted_and_target_observations_use_same_program_definitions=True,
        memory_grounding_only_on_rule_renewal=True,
        source_model='v43 best3500; no Qwen vision substitution',
        frozen_feature_probe='Original Qwen vision and language hidden features did not improve held-out training inverse dynamics.',
        runtime_action_selection='Unchanged NumPy exhaustive probability derivation from exported integer table')
    if rank==0:atomic(out/'manifest.json',manifest)
    wrapper=DDP(model,device_ids=[local],find_unused_parameters=False) if world>1 else model
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    began=time.time();best=-1
    anchor_indices=random.Random(411017).sample(range(len(corpus.rows)),512)
    anchor_features=torch.as_tensor(corpus.raw[anchor_indices].copy(),device='cuda')
    with torch.no_grad():anchor_codes=model.observe(anchor_features).argmax(-1)
    anchor_update=first-1
    if (out/'development.jsonl').exists():
        best=max([json.loads(line)['successes'] for line in (out/'development.jsonl').read_text().splitlines() if line.strip()]+[-1])
    for update in range(first,args.updates+1):
        tick=time.time();model.train();data=corpus.sample(rng,args.batch_size,args.branch_batch,'cuda')
        data['world_updates_encoder']=update%5==0
        optimizer.zero_grad(set_to_none=True)
        forward_cuda_rng=torch.cuda.get_rng_state()
        forward_cpu_rng=torch.get_rng_state()
        loss,metrics=wrapper(data)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; last COMMITTED checkpoint preserved')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient; checkpoint preserved')
        # Capture the FIRST actual anomalous DDP backward pass on both ranks.
        # Fixed-checkpoint random probes may miss an intermittent failure.
        case=out/f'backward_case_rank{rank}.pt'
        if float(norm)>1000 and not case.exists():
            temp=case.with_suffix('.tmp')
            torch.save(dict(model={n:p.detach().cpu() for n,p in model.state_dict().items()},
                data={n:v.detach().cpu() if torch.is_tensor(v) else v for n,v in data.items()},
                update=update,rank=rank,loss=float(loss.detach()),gradient_norm=float(norm),
                forward_cuda_rng=forward_cuda_rng.cpu(),forward_cpu_rng=forward_cpu_rng,
                model_config=model.export_config(),tf32=True),temp)
            temp.replace(case)
        optimizer.step();torch.cuda.synchronize()
        row={k:float(v) for k,v in metrics.items()};row.update(update=update,time=time.time(),seconds=time.time()-tick,gradient_norm=float(norm),rank=rank,
            memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,state='training',pid=os.getpid())
        if update%100==0:
            with torch.no_grad():new_codes=model.observe(anchor_features).argmax(-1)
            row.update(training_anchor_observation_change_rate=float((new_codes!=anchor_codes).float().mean()),
                anchor_images=len(anchor_indices),anchor_previous_update=anchor_update)
            anchor_codes=new_codes;anchor_update=update
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
                retain_evaluation_tables(out)
            if world>1:dist.barrier()
        if bool(stopping):break
    if rank==0:atomic(out/'status.json',dict(state='paused' if stop[0] else 'finished',update=update,time=time.time(),elapsed=time.time()-began))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
