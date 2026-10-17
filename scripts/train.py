import argparse,datetime,gc,json,os,random,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from nesy_vla.runtime import read,atomic,append,setup,DataBank,save_checkpoint,load_checkpoint,initialize_symbolic_modules,initialize_task_token_model
from nesy_vla.model import NavigationModel
from nesy_vla.policy import rollout,rollout_many,demonstration,demonstrations_many
from nesy_vla.training import advantages,train_chunk,train_batch,train_revision
from nesy_vla.revision import compare_candidates
from nesy_vla.batching import next_batch_size,pack_chunks
from nesy_vla.diagnostics import audit_policies


def sync_gradients(parameters,world):
    # Coalesced communication, with a fixed order even for unused branches.
    bucket=[];count=0
    def flush():
        if not bucket:return
        flat=torch.cat([p.grad.flatten() for p in bucket])
        if world>1:dist.all_reduce(flat);flat.div_(world)
        offset=0
        for p in bucket:
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
        bucket.clear()
    for p in parameters:
        if p.grad is None:p.grad=torch.zeros_like(p)
        if bucket and (bucket[0].dtype!=p.dtype or count+p.numel()>4_000_000):flush();count=0
        bucket.append(p);count+=p.numel()
    flush()


def backward_batches(model,optimizer,chunks,normalization,batch_size,out,rank,update):
    optimizer.zero_grad(set_to_none=True);statistics=[]
    limits=read(out/f'batch_tuning_rank{rank}.json') if (out/f'batch_tuning_rank{rank}.json').exists() else {}
    frame_budget=limits.get('frame_budget',model.cfg.get('training_frame_budget',130))
    revision_limit=limits.get('revision_limit',model.cfg.get('revision_chunk_limit',13))
    batches=pack_chunks(chunks,batch_size,frame_budget,revision_limit)
    try:
        for batch in batches:
            loss,metrics=train_batch(model,batch,normalization)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite joint loss')
            fraction=(sum(len(c['rows']) for c,_ in batch)/max(1,sum(len(c['rows']) for c,_ in chunks))
                      if model.cfg.get('action_normalized_loss',False) else len(batch)/max(1,len(chunks)))
            (loss*fraction).backward();statistics.extend(metrics)
            del loss,metrics
        return statistics,batch_size,max(map(len,batches),default=0),frame_budget,revision_limit
    except RuntimeError as error:
        if 'out of memory' not in str(error).lower():raise
        append(out/f'memory_backoff_rank{rank}.jsonl',{'update':update,'failed_batch':len(batch),
            'failed_frames':sum(len(c['rows']) for c,_ in batch),'error':str(error)[:300],
            'recovery':'fresh process from last full committed update'})
        atomic(out/f'batch_tuning_rank{rank}.json',{'update':update-1,'next_batch':max(1,batch_size-2),
            'frame_budget':max(10,frame_budget-20),'revision_limit':max(1,revision_limit-2)})
        # Non-reentrant checkpoint state was observed to remain invalid after an OOM.
        # A fresh worker restores optimizer/RNG and discards every partial local gradient.
        raise SystemExit(75) from error


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/navigation.json')
    parser.add_argument('--out',required=True);parser.add_argument('--updates',type=int)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    cfg=read(args.config)
    assert cfg['ppo_epochs']==1, 'This trainer performs one full on-policy pass per synchronized update'
    if args.updates:cfg['updates']=args.updates
    if args.smoke:cfg.update(rollouts_per_rank=1,demonstrations_per_rank=1,candidate_every=1,ppo_epochs=1)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    assert world<=8;torch.cuda.set_device(local);setup(cfg)
    if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=45))
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    bank=DataBank(cfg);model=NavigationModel(cfg).to('cuda').eval()
    if cfg.get('generator_backend')=='task_tokens':
        expected=sorted({'Find the '+e['category']+' and stop close to it.' for e in bank.train})
        assert sorted(cfg['task_vocabulary'])==expected,'Task vocabulary must come only from training instructions'
    params=[p for p in model.parameters() if p.requires_grad]
    groups=[]
    for prefix,lr in [('generator.vlm.',cfg['vlm_learning_rate']),('grounder.vision.',cfg['vision_learning_rate'])]:
        groups.append({'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefix)],'lr':lr})
    used={id(p) for g in groups for p in g['params']}
    groups.append({'params':[p for p in params if id(p) not in used],'lr':cfg['learning_rate']})
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    generator=random.Random(cfg['seed']+rank*1009);complete=0
    if (out/'latest/COMMITTED').exists():
        state=load_checkpoint(out/'latest',model,optimizer);complete=state['update']
        assert state['config']['revision_interval']==10 and state['config']['horizon']==32
        saved=state['rng_states'][rank];generator.setstate(saved['sampler']);random.setstate(saved['python'])
        np.random.set_state(saved['numpy']);torch.set_rng_state(saved['torch']);torch.cuda.set_rng_state(saved['cuda'])
    else:
        if cfg.get('initialize_execution_from'):
            from nesy_vla.runtime import initialize_execution_supervision
            assert not any(cfg.get(k) for k in ['initialize_from','initialize_symbolic_from','initialize_task_tokens_from','initialize_visual_from'])
            initialization=initialize_execution_supervision(cfg['initialize_execution_from'],model)
            if rank==0:atomic(out/'initialization.json',initialization)
        if cfg.get('initialize_task_tokens_from'):
            assert not cfg.get('initialize_from') and not cfg.get('initialize_symbolic_from')
            initialization=initialize_task_token_model(cfg['initialize_task_tokens_from'],model)
            if rank==0:atomic(out/'initialization.json',initialization)
        if cfg.get('initialize_visual_from'):
            from nesy_vla.visual_transfer import initialize_visual_evidence
            visual_initialization=initialize_visual_evidence(cfg['initialize_visual_from'],model)
            if rank==0:atomic(out/'visual_initialization.json',visual_initialization)
        if cfg.get('initialize_symbolic_from'):
            assert not cfg.get('initialize_from'),'Choose exact resume or symbolic-only initialization'
            initialization=initialize_symbolic_modules(cfg['initialize_symbolic_from'],model)
            if rank==0:atomic(out/'initialization.json',initialization)
        if cfg.get('initialize_from'):
            requested_lrs=[group['lr'] for group in optimizer.param_groups]
            state=load_checkpoint(cfg['initialize_from'],model,optimizer)
            for group,lr in zip(optimizer.param_groups,requested_lrs):group['lr']=lr
            if rank==0:atomic(out/'initialization.json',{'source':cfg['initialize_from'],
                'source_update':state['update'],'model_and_optimizer_restored':True,
                'learning_rates':requested_lrs,
                'new_run_updates_start_at':1,'sampler_rng':'new deterministic per-rank seeds for changed GPU count'})
        torch.manual_seed(cfg['seed']+rank*1009)
    # A synchronous target uses exactly the current grounded symbol alphabet,
    # including on the first update after transferring a lagged EMA checkpoint.
    # Target evaluation stays stop-gradient and action inference is unchanged.
    if cfg.get('teacher_decay',.995)==0:model.update_teacher(0.)
    if rank==0:
        atomic(out/'config.json',cfg);atomic(out/'data_provenance.json',bank.provenance)
        atomic(out/'architecture.json',{'trainable_parameters':sum(p.numel() for p in params),
            'all_actions_from_soft_logic':True,'program_revision_every_executed_actions':10,
            'maps':False,'handwritten_predicates':False,
            'full_training_components':(['BC','EMA transition','symbolic inverse','outcomes','unrolled program revision']
                if cfg.get('training_mode')=='supervised' else
                ['BC','PPO','EMA transition','symbolic inverse','outcomes','program revision distillation']),
            'balanced_transition_loss':cfg.get('balanced_transition_loss',False),
            'base_model':cfg['model_path'] if cfg.get('generator_backend') not in ['symbolic_gru','task_tokens'] else None,
            'generator_backend':cfg.get('generator_backend','vlm'),
            'vision_backbone':cfg.get('vision_backbone','convnext_tiny'),
            'frozen_vision':not any(p.requires_grad for p in model.grounder.vision.parameters()),
            'discrete_reasoning':cfg.get('discrete_reasoning',False),'world_size':world})
    simulators=[bank.simulator(out/f'sim_rank{rank}_{i}.log',local) for i in range(cfg['rollouts_per_rank'])]
    sim=simulators[0]
    effective_batch=cfg.get('training_chunk_batch',1)
    if (out/f'batch_tuning_rank{rank}.json').exists():
        effective_batch=read(out/f'batch_tuning_rank{rank}.json')['next_batch']
    try:
        for update in range(complete+1,cfg['updates']+1):
            started=time.time();model.eval();policies=[];results=[];episodes=[];trials=[]
            torch.cuda.reset_peak_memory_stats()
            supervised=cfg.get('training_mode')=='supervised'
            if not supervised:
                for i in range(cfg['rollouts_per_rank']):
                    episodes.append(bank.sample(generator,(update-1)*world*cfg['rollouts_per_rank']+rank*cfg['rollouts_per_rank']+i))
                policies,results=rollout_many(model,simulators,episodes,reward_overrides=cfg.get('training_reward_overrides'))
            for policy in policies:advantages(policy)
            rollout_seconds=time.time()-started
            # Real successful demonstrations make revision events available even when an untrained policy stops early.
            demo_ids=[]
            if cfg.get('recovery_demo_fraction',0)>0:
                from nesy_vla.resources import recovery_demonstration_schedule
                extras=set(bank.paired_demo_ids)|set(bank.recovery_demo_ids)
                demo_ids=recovery_demonstration_schedule([e for e in bank.demo_ids if e not in extras],
                    bank.paired_demo_groups,bank.recovery_demo_ids,update,rank,world,cfg['demonstrations_per_rank'],
                    cfg['seed'],cfg.get('paired_demo_fraction',0),cfg['recovery_demo_fraction'])
            elif cfg.get('paired_demo_fraction',0)>0:
                from nesy_vla.resources import paired_demonstration_schedule
                paired=set(bank.paired_demo_ids)
                demo_ids=paired_demonstration_schedule([e for e in bank.demo_ids if e not in paired],
                    bank.paired_demo_groups,update,rank,world,cfg['demonstrations_per_rank'],
                    cfg['seed'],cfg['paired_demo_fraction'])
            elif supervised:
                from nesy_vla.resources import demonstration_schedule
                demo_ids=demonstration_schedule(bank.demo_ids,update,rank,world,cfg['demonstrations_per_rank'],cfg['seed'])
            else:
                for i in range(cfg['demonstrations_per_rank']):
                    eligible=[e for e in bank.demo_ids if len(bank.demos[e])>10] if i==0 else bank.demo_ids
                    demo_ids.append(generator.choice(eligible))
            demos=demonstrations_many(model,bank,demo_ids)
            demo_seconds=time.time()-started-rollout_seconds
            symbolic_audit=audit_policies(model,policies+demos)
            candidates_started=time.time()
            if not supervised and update%cfg['candidate_every']==0:
                for episode,policy in zip(episodes,policies):
                    trial=compare_candidates(model,sim,bank,episode,policy,generator)
                    if trial is not None:trials.append(trial);break
                if not trials:
                    demo=demos[0]
                    # The demonstration episode ID is attached explicitly below, never inferred from an image.
                    trial=compare_candidates(model,sim,bank,bank.by_id[demo.episode_id],demo,generator)
                    if trial is not None:trials.append(trial)
            candidate_seconds=time.time()-candidates_started
            av=[r['advantage'] for p in policies for c in p.chunks for r in c['rows']]
            totals=torch.tensor([sum(av),sum(x*x for x in av),len(av)],device='cuda',dtype=torch.float64)
            if world>1:dist.all_reduce(totals)
            mean=totals[0]/totals[2].clamp_min(1);std=(totals[1]/totals[2].clamp_min(1)-mean**2).clamp_min(1e-6).sqrt()
            norm=(float(mean),float(std));stats=[]
            optimizer.zero_grad(set_to_none=True);model.train()
            chunks=[(c,False) for p in policies for c in p.chunks if c['rows']]
            chunks += [(c,True) for p in demos for c in p.chunks if c['rows']]
            # Accumulate gradients across bounded ten-action graphs, then one synchronized update.
            chunks.sort(key=lambda pair:(pair[0]['previous'] is not None,len(pair[0]['feedback'])))
            learning_started=time.time();requested_batch=effective_batch
            stats,effective_batch,actual_batch,frame_budget,revision_limit=backward_batches(
                model,optimizer,chunks,norm,effective_batch,out,rank,update)
            for trial in trials:
                loss=train_revision(model,trial)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite program revision loss')
                (loss/max(1,len(trials))).backward()
            sync_gradients(params,world)
            group_gradients={prefix:sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters()
                if p.requires_grad and p.grad is not None and n.startswith(prefix))**.5
                for prefix in ['generator.vlm.','generator.head.','generator.history.','grounder.vision.','grounder.query.','logic.']}
            gradient=torch.nn.utils.clip_grad_norm_(params,2.)
            if not torch.isfinite(gradient):raise FloatingPointError('Non-finite gradient')
            optimizer.step();model.update_teacher()
            learning_seconds=time.time()-learning_started
            torch.cuda.synchronize()
            free,total=torch.cuda.mem_get_info()
            external=max(0,total-free-torch.cuda.memory_reserved())
            peak=torch.cuda.max_memory_allocated()
            next_batch=next_batch_size(effective_batch,peak,external,total,len(chunks),
                cfg.get('maximum_chunk_batch',32),effective_batch!=requested_batch)
            atomic(out/f'batch_tuning_rank{rank}.json',{'update':update,'used_batch':effective_batch,
                'next_batch':next_batch,'torch_peak_gib':peak/2**30,'non_torch_gib':external/2**30,
                'frame_budget':frame_budget,'revision_limit':revision_limit,
                'estimated_physical_peak_gib':(peak+external)/2**30,'device_total_gib':total/2**30})
            rng={'sampler':generator.getstate(),'python':random.getstate(),'numpy':np.random.get_state(),
                 'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
            rng_states=[None]*world
            if world>1:dist.all_gather_object(rng_states,rng)
            else:rng_states=[rng]
            record={'update':update,'rank':rank,'seconds':time.time()-started,'episodes':len(results),
                'training_mode':cfg.get('training_mode','joint'),'demonstrations':len(demos),
                'demonstration_episode_ids':demo_ids,
                'successes':sum(bool(x.get('success')) for x in results),'max_actions':32,
                'revision_events':sum(max(0,len(p.chunks)-1) for p in policies+demos),
                'rollout_revision_events':sum(max(0,len(p.chunks)-1) for p in policies),
                'demonstration_revision_events':sum(max(0,len(p.chunks)-1) for p in demos),
                'candidate_trials':[{k:v for k,v in t.items() if k not in {'chunk','target'}} for t in trials],
                'symbolic_audit':symbolic_audit,
                'gradient_norm':float(gradient),'gradient_by_module':group_gradients,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30,
                'gpu_reserved_peak_gib':torch.cuda.max_memory_reserved()/2**30,'training_chunk_batch':actual_batch,
                'training_chunk_limit':effective_batch,
                'next_training_chunk_batch':next_batch,'training_chunks':len(chunks),
                'training_frames':sum(len(c['rows']) for c,_ in chunks),
                'phase_seconds':{'rollout':rollout_seconds,'demonstration':demo_seconds,
                                 'candidate_trials':candidate_seconds,'learning_and_sync':learning_seconds},
                'losses':{k:sum(s[k] for s in stats if k in s)/sum(k in s for s in stats)
                          for k in set().union(*(s.keys() for s in stats))},
                'episode_ids':[e['episode_id'] for e in episodes]}
            append(out/f'rank{rank}.jsonl',record)
            all_records=[None]*world
            if world>1:dist.all_gather_object(all_records,record)
            else:all_records=[record]
            if rank==0:
                save_checkpoint(out/'latest',model,optimizer,update,cfg,rng_states)
                if update%cfg['snapshot_every']==0 or update==cfg['updates']:
                    save_checkpoint(out/f'step_{update:06d}',model,optimizer,update,cfg,rng_states)
                atomic(out/'status.json',{'state':'training','update':update,'time':time.time(),'world_size':world,
                    'training_mode':cfg.get('training_mode','joint'),
                    'demonstration_episodes':sum(x['demonstrations'] for x in all_records),
                    'last_update_seconds':max(x['seconds'] for x in all_records),'revision_interval':10,'checkpoint':'latest',
                    'training_successes':sum(x['successes'] for x in all_records),
                    'training_episodes':sum(x['episodes'] for x in all_records)})
                append(out/'training_summary.jsonl',{'update':update,'episodes':sum(x['episodes'] for x in all_records),
                    'successes':sum(x['successes'] for x in all_records),
                    'seconds':max(x['seconds'] for x in all_records),
                    'rollout_revision_events':sum(x['rollout_revision_events'] for x in all_records),
                    'demonstration_revision_events':sum(x['demonstration_revision_events'] for x in all_records)})
                print(json.dumps(record),flush=True)
            effective_batch=next_batch
            if world>1:dist.barrier()
            # Checkpoint first. Rendering errors cannot discard an already completed update.
            from nesy_vla.reporting import save_trace
            for i,(policy,ep,metrics) in enumerate(zip(policies,episodes,results)):
                limit=cfg.get('training_trace_episodes_per_rank',len(policies))
                # Rotate diagnostic examples; all rollout outcomes and IDs remain
                # in rank logs, and every held-out evaluation retains full traces.
                if (i-update)%max(1,len(policies))>=limit:continue
                save_trace(out/f'traces/update_{update:06d}/rank{rank}_{i}',policy,ep,metrics)
            if rank==0 and (update==1 or update%cfg['candidate_every']==0):
                save_trace(out/f'demonstration_traces/update_{update:06d}',demos[0],bank.by_id[demos[0].episode_id],
                    {'source':'successful expert demonstration replay','not_autonomous_policy_success':True})
            if update%cfg['evaluation_every']==0:
                from nesy_vla.evaluation import evaluate_tasks,aggregate
                if rank==0:
                    # Preserve the exact evaluated weights even if later updates overwrite latest.
                    save_checkpoint(out/f'evaluations/step_{update:06d}/checkpoint',model,optimizer,
                        update,cfg,None,include_optimizer=False)
                if world>1:dist.barrier()
                model.eval()
                for suite,tasks in [('development',bank.dev),('fixed80',bank.fixed)]:
                    folder=out/f'evaluations/step_{update:06d}/{suite}'
                    rows=evaluate_tasks(model,simulators,tasks[rank::world],folder,stochastic=False)
                    atomic(folder/f'rank{rank}.json',rows)
                    if world>1:dist.barrier()
                    if rank==0:
                        all_rows=[r for n in range(world) for r in read(folder/f'rank{n}.json')]
                        atomic(folder/'summary.json',aggregate(all_rows))
                if rank==0:
                    development=read(out/f'evaluations/step_{update:06d}/development/summary.json')
                    previous=read(out/'best_score.json') if (out/'best_score.json').exists() else {'success_rate':-1}
                    if development['success_rate']>previous['success_rate']:
                        save_checkpoint(out/'best_development',model,optimizer,update,cfg,rng_states)
                        atomic(out/'best_score.json',dict(development,update=update))
                if world>1:dist.barrier()
        if rank==0:atomic(out/'status.json',{'state':'complete','update':cfg['updates'],'time':time.time()})
    finally:
        for sim in simulators:sim.close()
        if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
