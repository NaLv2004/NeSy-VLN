"""Train complete generated anonymous rules; task return is external supervision."""
import argparse,fcntl,gc,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.data import BASE,read,sha
from predictive_nesy.environment import make_bank
from predictive_nesy.anonymous_data import load_windows,batch,OutcomeBank,canonical_provenance
from predictive_nesy.anonymous_model import AnonymousRuleModel,Config
from predictive_nesy.anonymous_training import objective,outcome_objective
from predictive_nesy.anonymous_planning import loss as planning_objective


def atomic(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temporary,path)


def save(path,model,optimizer,step,manifest,rng):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),update=step,model_config=model.export_config(),
        manifest=manifest,random_state=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temporary,path);atomic(path.with_suffix('.COMMITTED.json'),dict(path=path.name,sha256=sha(path),update=step,time=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--initialize',required=True)
    p.add_argument('--updates',type=int,default=6000);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--eval-every',type=int,default=400);p.add_argument('--outcomes',action='append',default=[])
    p.add_argument('--outcome-batch-size',type=int,default=8);p.add_argument('--planning-batch-size',type=int,default=4)
    p.add_argument('--planning-weight',type=float,default=2.);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in {'3','4','5','7'}:raise RuntimeError('GPU not authorized')
    out=Path(a.out).resolve()
    if not out.is_relative_to(BASE/'predictive_rules_20261015'):raise ValueError('Output escaped isolated root')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(1528);rng=random.Random(1528);torch.backends.cuda.matmul.allow_tf32=True
    atomic(out/'status.json',dict(state='preparing',gpu=int(gpu),pid=os.getpid(),time=time.time()))
    original=torch.load(a.initialize,map_location='cpu',weights_only=False);old=original['manifest']
    bank=make_bank();windows,provenance=load_windows(old['data']['source'],bank.provenance['train_layouts'])
    if provenance!=canonical_provenance(old['data']):raise ValueError('Training split or corpus changed')
    labels=np.load(out.parent/'run_v1/learned_symbol_targets.npz')['labels']
    if len(labels)!=len(windows):raise ValueError('Learned symbol label mismatch')
    for w,y in zip(windows,labels):w['teacher_ids']=y.astype(np.int64)
    model=AnonymousRuleModel(Config(**original['model_config'])).cuda()
    transferable={'tokens','observation','memory_features','context_read','action_embed','task','history','context_norm',
        'definitions','parents','transitions','reconstruct','inverse'}
    inherited_anonymous=old.get('architecture','').startswith('anonymous_generated_rules_')
    current=model.state_dict();loaded=[]
    for name,value in original['model'].items():
        if (inherited_anonymous or name.split('.')[0] in transferable) and name in current and value.shape==current[name].shape:
            current[name]=value;loaded.append(name)
    if inherited_anonymous and set(loaded)!=set(current):raise ValueError('Incomplete anonymous checkpoint transfer')
    model.load_state_dict(current)
    banks=[OutcomeBank(path,bank.provenance['train_layouts']) for path in a.outcomes]
    manifest=dict(architecture='anonymous_generated_rules_v6',model=model.export_config(),data=provenance,
        frozen_vision=old['frozen_vision'],source_model=old['source_model'],
        initialize=dict(path=a.initialize,sha256=sha(a.initialize),update=original['update'],copied_parameters=loaded),
        generated_parameters=['definitions','parents','transitions','rewards','values','weights'],
        state_semantics='anonymous learned categories only',semantic_state_supervision=False,
        scalar_task_score='movement -0.02; successful terminal action +1; unsuccessful terminal action -1',
        scalar_score_role='external task return; never a symbolic variable or proposition',
        horizon=32,revision_interval=10,boundary_only_neural_inference=True,no_maps=True,
        planner='complete anonymous transition-table beam search; depth10 width32',
        reward_training='measured scalar returns only; planning-ranking gradients do not rewrite reward table entries',
        terminal_budget_bootstrap=False,initial_optimizer_restored=inherited_anonymous,
        outcome_banks=[dict(path=str(x.directory),**x.refresh()) for x in banks],
        train_args=vars(a),trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),created=time.time())
    if (out/'manifest.json').exists():
        existing=read(out/'manifest.json')
        if existing['data']!=provenance or existing['architecture']!=manifest['architecture']:raise ValueError('Run identity changed')
        manifest=existing
    else:atomic(out/'manifest.json',manifest)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);step=0
    if inherited_anonymous:
        optimizer.load_state_dict(original['optimizer'])
        for group in optimizer.param_groups:group['lr']=a.lr
        rng.setstate(original['random_state']);torch.set_rng_state(original['torch_rng'].cpu());torch.cuda.set_rng_state(original['cuda_rng'].cpu())
    del original,current;gc.collect();torch.cuda.empty_cache()
    if a.resume and (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step=saved['update']
        rng.setstate(saved['random_state']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());del saved
    starts=[w['start'] for w in windows];by_start={s:[w for w in windows if w['start']==s] for s in set(starts)}
    best=read(out/'best_development.json')['successes'] if (out/'best_development.json').exists() else -1
    began=time.time()
    while step<a.updates:
        begin=time.time();model.train();start=rng.choice(starts)
        data=batch(rng.choices(by_start[start],k=a.batch_size),'cuda');optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,data)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite supervised objective')
        loss.backward();del loss,data
        if step%100==0:
            for source in banks:source.refresh()
        available=[x for x in banks if x.keys]
        if available:
            source=rng.choice(available);data=source.sample(rng,a.outcome_batch_size,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=outcome_objective(model,data)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite outcome objective')
            loss.backward();metrics.update(extra);metrics['loss']+=float(loss.detach());metrics['outcome_source']=source.directory.name
            del loss,data
        if a.planning_weight:
            data=batch(rng.choices(by_start[start],k=a.planning_batch_size),'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=planning_objective(model,data)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sequence objective')
            (a.planning_weight*loss).backward();metrics.update(extra);metrics['loss']+=float(a.planning_weight*loss.detach());del loss,data
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();step+=1;torch.cuda.synchronize()
        metrics.update(update=step,time=time.time(),seconds=time.time()-begin,elapsed_seconds=time.time()-began,
            grad_norm=float(norm),memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,boundary=start)
        with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        atomic(out/'status.json',dict(state='training',pid=os.getpid(),gpu=int(gpu),**metrics))
        if step==1 or step%20==0:print(json.dumps(metrics),flush=True)
        if step%100==0 or step==a.updates:save(out/'latest.pt',model,optimizer,step,manifest,rng)
        if step%a.eval_every==0:
            from evaluate import evaluate_generated
            atomic(out/'status.json',dict(state='evaluating_development',pid=os.getpid(),gpu=int(gpu),update=step,time=time.time()))
            result=evaluate_generated(model,manifest,bank,out/f'development_step_{step:06d}','development');result['update']=step
            with (out/'validation.jsonl').open('a') as stream:stream.write(json.dumps(result,allow_nan=False)+'\n')
            if result['successes']>best:
                best=result['successes'];save(out/'best_development.pt',model,optimizer,step,manifest,rng);atomic(out/'best_development.json',result)
    atomic(out/'status.json',dict(state='training_budget_complete',pid=os.getpid(),gpu=int(gpu),update=step,time=time.time(),goal_complete=False))


if __name__=='__main__':main()
