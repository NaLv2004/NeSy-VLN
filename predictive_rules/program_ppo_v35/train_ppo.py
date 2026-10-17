"""On-policy semi-Markov PPO for hard-symbolic program generation."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
import distributed as dd
from torch.nn import functional as F
from program_policy import ProgramPolicy, ProgramValue, Config, replay_batch, normal_logp, exact_kl
from rollout import Collector
from predictive_nesy.environment import make_bank
from predictive_nesy.data import sha, read
from evaluate_state512_v25 import atomic


def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def committed_file(path):
    commit=read(path.with_suffix('.COMMITTED.json'))
    result=path.parent/commit['filename']
    assert result.parent.resolve()==path.parent.resolve()
    assert sha(result)==commit['sha256']
    return result


def save(path,model,critic,optimizer,value_optimizer,update,manifest,rng,best):
    rngs=dd.gather(dict(python=rng.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
    if dd.rank()!=0:
        dd.barrier();return
    marker=path.with_suffix('.COMMITTED.json')
    previous=read(marker) if marker.exists() else {}
    final=path.with_name(f'{path.stem}_{update:06d}_{time.time_ns()}.pt')
    temporary=final.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),critic=critic.state_dict(),model_config=model.export_config(),
        optimizer=optimizer.state_dict(),value_optimizer=value_optimizer.state_dict(),update=update,manifest=manifest,
        rank_rng=rngs,best=best),temporary)
    with temporary.open('rb') as f:os.fsync(f.fileno())
    os.replace(temporary,final)
    atomic(marker,dict(update=update,sha256=sha(final),filename=final.name,
                      previous_filename=previous.get('filename'),time=time.time()))
    # Retain the current and previous committed file, plus separately saved best.
    if previous.get('previous_filename'):
        obsolete=path.parent/previous['previous_filename']
        assert obsolete.parent.resolve()==path.parent.resolve() and obsolete.name.startswith(path.stem+'_')
        obsolete.unlink(missing_ok=True)
    dd.barrier()


def update_policy(model,critic,optimizer,value_optimizer,segments,std,epochs=4):
    # cuDNN GRU needs training mode for backward; these branches have no dropout.
    model.train()
    n=int(dd.total(len(segments)))
    advantage=torch.tensor([s['advantage'] for s in segments],device='cuda')
    returns=torch.tensor([s['return'] for s in segments],device='cuda')
    moments=dd.total([float(advantage.sum()),float(advantage.square().sum())])
    average=float(moments[0])/n;deviation=max(1e-12,float(moments[1])/n-average**2)**.5
    advantage=(advantage-average)/max(deviation,1e-6)
    batches=[(begin,replay_batch(segments[begin:begin+16])) for begin in range(0,len(segments),16)]
    actor_parameters=[p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        replay_error=max(float((normal_logp(b['sample'],model.context_mean(b['task'],b['previous'],b['lengths'],b['visual'],b['base']),std)-b['logp']).abs().max()) for _,b in batches)
    replay_error=float(dd.maximum(replay_error))
    if replay_error>.002:raise RuntimeError(f'On-policy program likelihood replay differs: {replay_error}')
    accepted=0;rejected=0;gradients=[];losses=[];max_kl=0.;clip_fraction=0.
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        before=[p.detach().clone() for p in actor_parameters]
        before_optimizer=copy.deepcopy(optimizer.state_dict())
        total=0.
        for begin,b in batches:
            mean=model.context_mean(b['task'],b['previous'],b['lengths'],b['visual'],b['base'])
            logp=normal_logp(b['sample'],mean,std)
            ratio=(logp-b['logp']).clamp(-20,20).exp()
            a=advantage[begin:begin+len(mean)]
            loss=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).sum()/n
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO objective')
            loss.backward();total+=float(loss.detach())
        dd.synchronize_gradients(actor_parameters)
        norm=torch.nn.utils.clip_grad_norm_(actor_parameters,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite program gradient')
        optimizer.step()
        with torch.no_grad():
            divergences=[];clipped=[]
            for _,b in batches:
                mean=model.context_mean(b['task'],b['previous'],b['lengths'],b['visual'],b['base'])
                divergences.append(exact_kl(b['mean'],mean,std))
                ratio=(normal_logp(b['sample'],mean,std)-b['logp']).exp()
                clipped.append((ratio.sub(1).abs()>.2).float())
            divergence=torch.cat(divergences)
            kl=float(dd.total(divergence.sum()))/n;maximum=float(dd.maximum(divergence.max()))
            clip_fraction=float(dd.total(torch.cat(clipped).sum()))/n
        if not np.isfinite(kl) or kl>.10:
            with torch.no_grad():
                for p,previous in zip(actor_parameters,before):p.copy_(previous)
            optimizer.load_state_dict(before_optimizer)
            for group in optimizer.param_groups:group['lr']=max(1e-7,group['lr']*.5)
            rejected+=1
            continue
        accepted+=1;gradients.append(float(norm));losses.append(total);max_kl=maximum
        if kl>.05:break
    # Separate training-only critic: no gradients can reach rules or perception.
    values=torch.cat([s['value_input'] for s in segments]).cuda()
    value_loss=0.
    for _ in range(4):
        value_optimizer.zero_grad(set_to_none=True)
        prediction=critic(values)
        loss=.5*F.mse_loss(prediction,returns,reduction='sum')/n
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite critic')
        loss.backward();dd.synchronize_gradients(list(critic.parameters()));torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);value_optimizer.step()
        value_loss=float(loss)
    with torch.no_grad():
        kls=[];changes=[]
        for _,b in batches:
            mean=model.context_mean(b['task'],b['previous'],b['lengths'],b['visual'],b['base'])
            kls.append(exact_kl(b['mean'],mean,std))
            changes.append((mean-b['mean']).square().mean(-1))
    model.eval()
    return dict(program_decisions=len(segments),ppo_loss=float(dd.total(float(np.mean(losses)) if losses else 0.)),
        critic_loss=float(dd.total(value_loss)),actor_steps_accepted=accepted,actor_steps_rejected=rejected,
        actor_gradient_norm=float(np.mean(gradients)) if gradients else 0.,
        program_kl=float(dd.total(torch.cat(kls).sum()))/n,program_max_kl=max_kl,
        program_context_rms_change=(float(dd.total(torch.cat(changes).sum()))/n)**.5,
        clip_fraction=clip_fraction,likelihood_replay_max_error=replay_error,
        advantage_std=deviation,
        actor_lr=optimizer.param_groups[0]['lr'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--parallel',type=int,default=16);p.add_argument('--episodes',type=int,default=32)
    p.add_argument('--eval-every',type=int,default=10);p.add_argument('--updates',type=int,default=0)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    dd.initialize();local=out/f'rank{dd.rank()}';local.mkdir(exist_ok=True)
    assert str(out.resolve()).startswith('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_program_ppo_v35')
    lock=(local/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(351717);rng=random.Random(351717+1009*dd.rank())
    source=Path(args.source);assert sha(source)==read(source.with_suffix('.COMMITTED.json'))['sha256']
    source_state=torch.load(source,map_location='cpu',weights_only=False)
    model=ProgramPolicy(Config(**source_state['model_config'])).cuda().eval()
    model.load_state_dict(source_state['model']);parameters=model.configure_ppo()
    critic=ProgramValue().cuda();optimizer=torch.optim.Adam(parameters,lr=1e-5)
    value_optimizer=torch.optim.Adam(critic.parameters(),lr=1e-3)
    bank=make_bank()
    train_ids={e['episode_id'] for e in bank.train};dev_ids={e['episode_id'] for e in bank.dev};fixed_ids={e['episode_id'] for e in bank.fixed}
    assert train_ids.isdisjoint(dev_ids|fixed_ids)
    assert len(bank.dev)==80
    frozen_cfg=read(bank.env['frozen_config'])
    manifest=dict(architecture='hard_program_context_ppo_v35',source=str(source),source_sha256=sha(source),
        source_architecture=source_state['manifest']['architecture'],data=bank.provenance,
        frozen_vision=source_state['manifest']['frozen_vision'],context_std=.1,generation_cost=.2,
        planning_failure_penalty=float(frozen_cfg['rewards']['false_stop']),
        budget_failure_penalty=float(frozen_cfg['budget_failure_penalty']),
        rewards=frozen_cfg['rewards'],training_episodes=len(bank.train),development_episode_ids=sorted(dev_ids),
        max_actions=32,parallel_simulators_per_gpu=args.parallel,episodes_per_gpu=args.episodes,
        world_size=dd.world(),episodes_per_update=args.episodes*dd.world(),eval_every=args.eval_every,
        actor_lr=1e-5,critic_lr=1e-3,ppo_epochs=4,clip=.2,target_kl=.05,maximum_kl=.1,
        gamma=.99,gae_lambda=.95,policy_decision='Two128-dimensional Gaussian rule/goal contexts per program revision',
        likelihood='Sum log density across all256 sampled coordinates; exact joint Gaussian KL',
        trainable_names=[n for n,v in model.named_parameters() if v.requires_grad],
        trainable_parameters=sum(v.numel() for v in parameters),
        fixed_modules=['visual encoder','anonymous definitions','complete rule decoder','goal decoder','correction decoder','renewal GRU'],
        symbolic_execution='Every action independently verified on a backward path to generated hard goal; all T/C transitions argmax',
        state_input='Single anonymous observed ID per executed action, unchanged existing interface',
        map_or_geometry_to_policy=False,supervised_action_loss=False,dynamics_supervision=False,
        critic_used_at_execution=False,neural_action_fallback=False,
        evaluation_mode='Deterministic context means and original hard proof selection',
        five_step_average_requirement='Measure actual executed actions divided by all programs including initial; not claimed guaranteed',
        storage='Rolling latest complete checkpoint plus best qualifying development checkpoint; rollout traces retained',
        automatic_stop_updates=args.updates)
    if dd.rank()==0:
        if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
        else:atomic(out/'manifest.json',manifest)
    dd.barrier();torch.manual_seed(351717+1009*dd.rank())
    del source_state
    update=0;best=-1;stop={'requested':False};collector=None
    signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True))
    signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True))
    if (out/'latest.COMMITTED.json').exists():
        saved=torch.load(committed_file(out/'latest.pt'),map_location='cuda',weights_only=False)
        assert saved['manifest']==manifest
        model.load_state_dict(saved['model']);critic.load_state_dict(saved['critic'])
        optimizer.load_state_dict(saved['optimizer']);value_optimizer.load_state_dict(saved['value_optimizer'])
        update=saved['update'];best=saved['best'];random_state=saved['rank_rng'][dd.rank()];rng.setstate(random_state['python'])
        torch.set_rng_state(random_state['torch'].cpu());torch.cuda.set_rng_state(random_state['cuda'].cpu());del saved
    atomic(local/'status.json',dict(state='initializing',pid=os.getpid(),update=update,time=time.time()))
    frozen_parameters={n:v.detach().clone() for n,v in model.named_parameters() if not v.requires_grad}
    try:
        collector=Collector(model,critic,bank,manifest,local,args.parallel)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        began=time.time()
        while not stop['requested'] and (not args.updates or update<args.updates):
            update+=1;t=time.time()
            episodes=[bank.sample(rng,(update-1)*args.episodes*dd.world()+dd.rank()*args.episodes+i) for i in range(args.episodes)]
            assert all(e['episode_id'] in train_ids for e in episodes)
            def progress(fields):
                atomic(local/'status.json',dict(state='collecting',pid=os.getpid(),update=update,**fields))
            segments,rollout=collector.collect(episodes,local/'rollouts'/f'update_{update:06d}',True,progress)
            collect_seconds=time.time()-t;optimized=time.time()
            with collector.gate.boundary(0):
                losses=update_policy(model,critic,optimizer,value_optimizer,segments,manifest['context_std'])
            # No uncounted changes to the deterministic policy following a sampled program.
            assert all(torch.equal(value,frozen_parameters[name]) for name,value in model.named_parameters() if not value.requires_grad)
            torch.cuda.synchronize()
            row=dict(**{**rollout,**losses},update=update,time=time.time(),collection_seconds=collect_seconds,
                optimization_seconds=time.time()-optimized,elapsed_seconds=time.time()-began,
                memory_peak_gib=torch.cuda.max_memory_allocated()/2**30)
            # Collection's own seconds is already present in rollout; explicit total wins.
            row['seconds']=time.time()-t
            append(local/'train.jsonl',row)
            global_row=dd.aggregate(dd.gather(row));global_row['update']=update
            if dd.rank()==0:append(out/'train.jsonl',global_row);print(json.dumps(global_row),flush=True)
            save(out/'latest.pt',model,critic,optimizer,value_optimizer,update,manifest,rng,best)
            atomic(local/'status.json',dict(state='training',pid=os.getpid(),**row))
            if dd.rank()==0:atomic(out/'status.json',dict(state='training',pid=os.getpid(),**global_row))
            del segments
            stop['requested']=bool(dd.maximum(int(stop['requested'])))
            if update%args.eval_every==0 and not stop['requested']:
                checkpoint_sha=read(out/'latest.COMMITTED.json')['sha256']
                def validation_progress(fields):
                    atomic(local/'status.json',dict(state='scheduled_development',pid=os.getpid(),update=update,**fields))
                discarded,result=collector.collect(bank.dev[dd.rank()::dd.world()],local/'development'/f'update_{update:06d}',False,validation_progress)
                del discarded
                result=dd.aggregate(dd.gather(result))
                result.update(update=update,checkpoint_sha256=checkpoint_sha,time=time.time())
                if dd.rank()==0:append(out/'validation.jsonl',result)
                if result['five_step_average_requirement_met'] and result['successes']>best:
                    best=result['successes'];save(out/'best.pt',model,critic,optimizer,value_optimizer,update,manifest,rng,best)
                if dd.rank()==0:print(json.dumps(dict(stage='development_result',**result)),flush=True)
            stop['requested']=bool(dd.maximum(int(stop['requested'])))
        save(out/'latest.pt',model,critic,optimizer,value_optimizer,update,manifest,rng,best)
        atomic(local/'status.json',dict(state='paused' if stop['requested'] else 'budget_complete',update=update,pid=os.getpid(),time=time.time()))
        if dd.rank()==0:atomic(out/'status.json',read(local/'status.json'))
    finally:
        if collector is not None:collector.close()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import sys
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
            local=out/f'rank{os.environ.get("RANK","0")}';local.mkdir(exist_ok=True)
            atomic(local/'error.json',dict(time=time.time(),pid=os.getpid(),error=repr(error),traceback=traceback.format_exc()))
        raise
