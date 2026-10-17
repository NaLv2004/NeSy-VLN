"""Train the complete observation-gated predictive-memory system on GPU 7."""
import argparse,fcntl,gc,json,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.data import BASE,read,sha,load_training_windows,batch_windows
from predictive_nesy.environment import load_previous,make_bank
from predictive_nesy.model_memory import PredictiveMemoryModel
from predictive_nesy.training_memory import objective
from train import atomic,save_checkpoint,learned_symbol_targets


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--updates',type=int,default=6000)
    p.add_argument('--eval-every',type=int,default=400);p.add_argument('--resume',action='store_true')
    p.add_argument('--outcomes');p.add_argument('--cf-batch-size',type=int,default=8)
    p.add_argument('--cf-weight',type=float,default=1.);p.add_argument('--warm-start')
    p.add_argument('--extra-outcomes')
    p.add_argument('--planning-weight',type=float,default=0.)
    p.add_argument('--planning-batch-size',type=int,default=4)
    p.add_argument('--planning-beam-width',type=int,default=32)
    args=p.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in {'3','4','5','7'}:raise RuntimeError('Use one explicitly authorized GPU: 3, 4, 5 or 7')
    out=Path(args.out).resolve()
    if not out.is_relative_to(BASE/'predictive_rules_20261015'):raise ValueError('Output escaped experiment root')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(1511);rng=random.Random(1511)
    torch.backends.cuda.matmul.allow_tf32=True
    atomic(out/'status.json',dict(state='preparing',pid=os.getpid(),gpu=int(gpu),time=time.time()))
    teacher,old_manifest,old_commit=load_previous();bank=make_bank()
    assert bank.provenance==old_manifest['data']
    windows,provenance=load_training_windows(BASE/'perceptual_sequences_v73/tail_1e4',bank.provenance['train_layouts'])
    # Reuse immutable training targets; do not regenerate existing evaluations.
    prior=out.parent/'run_v1'
    assert read(prior/'manifest.json')['source_model']==old_commit
    assert read(prior/'manifest.json')['data']==provenance
    if not (out/'learned_symbol_targets.npz').exists():
        shutil.copyfile(prior/'learned_symbol_targets.npz',out/'learned_symbol_targets.npz')
    learned_symbol_targets(teacher,windows,out)
    model=PredictiveMemoryModel().cuda()
    for name in ['tokens','observation','context_read','history','task','inverse']:
        getattr(model,name).load_state_dict(getattr(teacher,name).state_dict())
    del teacher;gc.collect();torch.cuda.empty_cache()
    counterfactual=None
    if args.outcomes:
        from predictive_nesy.counterfactual import CounterfactualBank
        counterfactual=CounterfactualBank(args.outcomes,bank.provenance['train_layouts'])
        if not counterfactual.keys:raise ValueError('No verified training outcomes yet')
    extra_counterfactual=None
    if args.extra_outcomes:
        from predictive_nesy.counterfactual import CounterfactualBank
        extra_counterfactual=CounterfactualBank(args.extra_outcomes,bank.provenance['train_layouts'])
    manifest=dict(architecture='generated_predictive_memory_rules_v2',model=model.export_config(),
        source_model=old_commit,data=provenance,frozen_vision=old_manifest['source_commit'],
        baseline_directory=str(prior/'baseline_1097'),horizon=32,revision_interval=10,
        boundary_only_neural_inference=True,no_maps=True,
        state='4 learned visual categories plus 4 learned predictive-memory categories; categorical beliefs propagated by rules',
        memory_semantics='No manually assigned meaning or hardcoded counter; targets learned from causal observation/action history',
        training='10-step open-loop prediction; future grounding is supervision only; same-variable parent correlation retained',
        planner='10-step categorical-belief beam search over complete generated rule tables; no neural actor',
        checkpoint_selection='development only',train_args=vars(args),created=time.time())
    if counterfactual is not None:
        manifest['counterfactual_training']=dict(path=args.outcomes,**counterfactual.refresh(),
            policy_access='labels only; no future observations replace propagated states',
            labels='successor visual categories, collision, goal eligibility; no maps, coordinates or distances')
    if args.planning_weight:
        manifest['multi_step_alignment']=dict(weight=args.planning_weight,
            batch_size=args.planning_batch_size,beam_width=args.planning_beam_width,
            horizon='same as deployed block, min(10,32-start)',
            objective='margin between complete demonstrated path and a searched alternative with a different first action',
            risk_gradient='detached; collision tables retain measured-outcome supervision')
    if extra_counterfactual is not None:
        manifest['additional_outcomes']=dict(path=args.extra_outcomes,**extra_counterfactual.refresh(),
            selection='equal probability between available demonstration and own-policy outcome banks',
            policy_access='training labels only; no within-block observation correction')
    if args.warm_start:
        saved=torch.load(args.warm_start,map_location='cuda',weights_only=False)
        if saved['manifest']['data']!=provenance:raise ValueError('Warm-start corpus changed')
        model.load_state_dict(saved['model'])
        manifest['warm_start']=dict(path=args.warm_start,sha256=sha(args.warm_start),update=saved['update'],optimizer='fresh')
        del saved
    if (out/'manifest.json').exists():
        manifest=read(out/'manifest.json');assert manifest['data']==provenance
    else:atomic(out/'manifest.json',manifest)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    update=0
    if args.resume and (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);update=saved['update']
        rng.setstate(saved['random_state']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        del saved
    by_start={s:[w for w in windows if w['start']==s] for s in sorted({w['start'] for w in windows})}
    starts=[w['start'] for w in windows]
    best=read(out/'best_development.json')['successes'] if (out/'best_development.json').exists() else -1
    started=time.time()
    while update<args.updates:
        begin=time.time();model.train()
        start=rng.choice(starts);batch=batch_windows(rng.choices(by_start[start],k=args.batch_size),'cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,batch)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; committed weights preserved')
        loss.backward()
        if counterfactual is not None:
            from predictive_nesy.counterfactual import objective as outcome_objective
            if update%100==0:
                metrics['cf_data']=counterfactual.refresh()
                if extra_counterfactual is not None:metrics['extra_cf_data']=extra_counterfactual.refresh()
            available=[counterfactual]
            if extra_counterfactual is not None and extra_counterfactual.keys:available.append(extra_counterfactual)
            source=rng.choice(available)
            metrics['cf_source']='own_policy' if source is extra_counterfactual else 'demonstration'
            cf_batch=source.sample(rng,args.cf_batch_size,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):cf_loss,cf_metrics=outcome_objective(model,cf_batch)
            if not torch.isfinite(cf_loss):raise FloatingPointError('Nonfinite counterfactual loss')
            (args.cf_weight*cf_loss).backward();metrics.update(cf_metrics)
            metrics['loss']=float(loss.detach()+args.cf_weight*cf_loss.detach())
            del cf_loss,cf_batch
        if args.planning_weight:
            from predictive_nesy.planning_alignment import loss as planning_objective
            chosen=rng.choices(by_start[start],k=args.planning_batch_size)
            planning_batch=batch_windows(chosen,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                planning_loss,planning_metrics=planning_objective(model,planning_batch,
                    beam_width=args.planning_beam_width)
            if not torch.isfinite(planning_loss):raise FloatingPointError('Nonfinite multi-step planning loss')
            (args.planning_weight*planning_loss).backward()
            metrics.update(planning_metrics)
            metrics['loss']+=float(args.planning_weight*planning_loss.detach())
            del planning_loss,planning_batch
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; committed weights preserved')
        optimizer.step();update+=1;torch.cuda.synchronize()
        metrics.update(update=update,seconds=time.time()-begin,time=time.time(),elapsed_seconds=time.time()-started,
            grad_norm=float(norm),memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,batch_size=args.batch_size,boundary=start)
        with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        atomic(out/'status.json',dict(state='training',pid=os.getpid(),gpu=int(gpu),**metrics))
        if update==1 or update%20==0:print(json.dumps(metrics),flush=True)
        if update%100==0 or update==args.updates:save_checkpoint(out/'latest.pt',model,optimizer,update,manifest,rng)
        if update%args.eval_every==0:
            from evaluate import evaluate_generated
            atomic(out/'status.json',dict(state='evaluating_development',update=update,pid=os.getpid(),gpu=int(gpu),time=time.time()))
            result=evaluate_generated(model,manifest,bank,out/f'development_step_{update:06d}','development');result['update']=update
            with (out/'validation.jsonl').open('a') as stream:stream.write(json.dumps(result,allow_nan=False)+'\n')
            if result['successes']>best:
                best=result['successes'];save_checkpoint(out/'best_development.pt',model,optimizer,update,manifest,rng)
                atomic(out/'best_development.json',result)
    atomic(out/'status.json',dict(state='training_budget_complete',update=update,pid=os.getpid(),gpu=int(gpu),time=time.time(),goal_complete=False))


if __name__=='__main__':main()
