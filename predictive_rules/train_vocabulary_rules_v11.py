"""Learn exact goal-directed rules with an automatically learned vocabulary."""
import argparse,fcntl,gc,json,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from predictive_nesy.data import BASE,read,sha
from predictive_nesy.environment import make_bank
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank,canonical_provenance
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel as SubgoalRuleModel,Config
from predictive_nesy.rollout_proof_training import objective,transition_objective


def atomic(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temporary,path)


def save(path,model,optimizer,step,manifest,random_states):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),update=step,model_config=model.export_config(),
        manifest=manifest,rank_random_states=random_states),temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temporary,path);atomic(path.with_suffix('.COMMITTED.json'),dict(path=path.name,sha256=sha(path),update=step,time=time.time()))


class TrainingModule(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,data,kind):return objective(self.model,data) if kind=='demonstrations' else transition_objective(self.model,data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--initialize',required=True)
    p.add_argument('--updates',type=int,default=6000);p.add_argument('--batch-size',type=int,default=48)
    p.add_argument('--eval-every',type=int,default=400);p.add_argument('--transitions',action='append',default=[])
    p.add_argument('--transition-batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if set(visible)-{'3','4','5','7'} or '7' in visible or len(visible)!=world:raise RuntimeError('Training GPUs must be authorized; GPU7 reserved for evaluation')
    torch.cuda.set_device(local);device=f'cuda:{local}';torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    out=Path(a.out).resolve()
    if not out.is_relative_to(BASE/'predictive_rules_20261015'):raise ValueError('Output escaped isolated root')
    out.mkdir(parents=True,exist_ok=True)
    if rank==0:
        lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(out/'status.json',dict(state='preparing',gpus=visible,pid=os.getpid(),time=time.time()))
    torch.manual_seed(1821);rng=random.Random(1821+rank);torch.backends.cuda.matmul.allow_tf32=True
    commit=read(Path(a.initialize).with_suffix('.COMMITTED.json'))
    if sha(a.initialize)!=commit['sha256']:raise ValueError('Initialization checkpoint hash mismatch')
    original=torch.load(a.initialize,map_location='cpu',weights_only=False);old=original['manifest'];bank=make_bank()
    windows,provenance=load_windows(old['data']['source'],bank.provenance['train_layouts'])
    if provenance!=canonical_provenance(old['data']):raise ValueError('Training split or corpus changed')
    model=SubgoalRuleModel(Config()).to(device)
    current=model.state_dict();loaded=[]
    for name,value in original['model'].items():
        preserve=True  # Full checkpoint initialization; freeze representation below.
        if preserve and name in current and value.shape==current[name].shape:current[name]=value;loaded.append(name)
    model.load_state_dict(current)
    frozen_prefixes=('transitions.','subgoal.')
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(frozen_prefixes))
    frozen_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if not name.startswith(frozen_prefixes)}
    sources=[TransitionBank(path,bank.provenance['train_layouts']) for path in a.transitions]
    manifest=dict(architecture='anonymous_rollout_proof_subgoal_v8',model=model.export_config(),data=provenance,
        frozen_vision=old['frozen_vision'],source_model=old['source_model'],
        initialize=dict(path=a.initialize,sha256=commit['sha256'],update=original['update'],copied_parameters=loaded),
        generated_parameters=['definitions','transitions','subgoal_logits'],
        state_semantics='anonymous learned categories only',semantic_state_supervision=False,
        reward_table=False,value_table=False,direct_action_head=False,initial_optimizer_restored=False,
        subgoal_target='learned encoding of demonstrated state after min(10, remaining actions); future RGB is supervision only',
        goal_acceptance='exact equality of every categorical factor; unreachable goals fail without fallback',
        horizon=32,revision_interval=10,boundary_only_neural_inference=True,no_maps=True,
        proof_action_supervision=True,learned_joint_state_count=128,
        proof_state_source='initial state plus exact rules and demonstrated action prefix; no future image premises',
        exact_amp_transition_lookup=True,
        training_uses_exact_argmax_transitions=True,
        visual_state_labels_ignore_future_actions=True,teacher_forced_dynamics_supervision=True,
        repeated_image_leaves_state_unchanged=True,visual_history_grounding_only_at_rule_boundaries=True,
        initialization_scope='retained learned image/history features, full neural definitions distilled from unsupervised visual clusters, fresh rule and goal heads',
        definition_initialization=old['definition_initialization'],
        world_model_intervention='stable automatically learned visual vocabulary, no semantic state labels or runtime clustering file',
        transition_sampling='bank probability proportional to its number of valid states; uniform state marginal within a bank',
        trainable_modules=['transitions','subgoal'],state_representation_frozen=True,
        planner='complete finite-state derivation to an anonymous goal; maximum depth10',
        transition_banks=[dict(path=str(x.directory),**x.refresh()) for x in sources],
        train_args=vars(a),world_size=world,global_batch_size=world*a.batch_size,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),created=time.time())
    if (out/'manifest.json').exists():
        existing=read(out/'manifest.json')
        if existing['data']!=provenance or existing['architecture']!=manifest['architecture']:raise ValueError('Run identity changed')
        manifest=existing
    elif rank==0:atomic(out/'manifest.json',manifest)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4);step=0
    del original,current;gc.collect();torch.cuda.empty_cache()
    if a.resume and (out/'latest.pt').exists():
        if sha(out/'latest.pt')!=read(out/'latest.COMMITTED.json')['sha256']:raise ValueError('Resume checkpoint hash mismatch')
        saved=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step=saved['update']
        state=saved['rank_random_states'][rank];rng.setstate(state['python']);torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state(state['cuda'].cpu());del saved
    wrapper=TrainingModule(model)
    if world>1:wrapper=DDP(wrapper,device_ids=[local],find_unused_parameters=True)
    starts=[w['start'] for w in windows];by_start={s:[w for w in windows if w['start']==s] for s in set(starts)}
    began=time.time();evaluation=None;eval_path=None;eval_step=None;best=-1
    if (out/'best_development.json').exists():best=read(out/'best_development.json')['successes']

    def collect_evaluation(wait=False):
        nonlocal evaluation,best
        if evaluation is None:return
        if wait:evaluation.wait()
        if evaluation.poll() is None:return
        if evaluation.returncode or not (eval_path/'summary.json').exists():
            atomic(out/'evaluation_error.json',dict(update=eval_step,returncode=evaluation.returncode,path=str(eval_path),time=time.time()))
        else:
            result=read(eval_path/'summary.json')
            with (out/'validation.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
            if result['successes']>best:
                best=result['successes'];atomic(out/'best_development.json',dict(**result,checkpoint=str(out/f'checkpoint_{eval_step:06d}.pt')))
        evaluation=None

    while step<a.updates:
        begin=time.time();model.train();start=rng.choice(starts);optimizer.zero_grad(set_to_none=True)
        data=batch(rng.choices(by_start[start],k=a.batch_size),device)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=wrapper(data,'demonstrations')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite state supervision')
        loss.backward();del loss,data
        available=[x for x in sources if x.keys]
        if available:
            source=rng.choices(available,weights=[len(x.keys) for x in available],k=1)[0]
            data=source.sample(rng,a.transition_batch_size,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=wrapper(data,'transitions')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite transition supervision')
            loss.backward();metrics.update(extra);metrics['loss']+=float(loss.detach());del loss,data
            for index,candidate in enumerate(sources):metrics[f'transition_bank_{index}_selected']=float(candidate is source)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();step+=1;torch.cuda.synchronize()
        names=sorted(metrics);values=torch.tensor([metrics[k] for k in names],device=device)
        if world>1:dist.all_reduce(values);values/=world
        metrics=dict(zip(names,values.tolist()))
        metrics.update(update=step,time=time.time(),seconds=time.time()-begin,elapsed_seconds=time.time()-began,
            grad_norm=float(norm),memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,global_batch_size=a.batch_size*world)
        if rank==0:
            with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            atomic(out/'status.json',dict(state='training',pid=os.getpid(),gpus=visible,**metrics));collect_evaluation()
            if step==1 or step%20==0:print(json.dumps(metrics),flush=True)
        if step==1 or step%100==0 or step==a.updates:
            state=dict(python=rng.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state());states=[None]*world
            if world>1:dist.all_gather_object(states,state)
            else:states=[state]
            if rank==0:save(out/'latest.pt',model,optimizer,step,manifest,states)
        if step%a.eval_every==0:
            if rank==0:
                collect_evaluation(wait=True)
                checkpoint=out/f'checkpoint_{step:06d}.pt';save(checkpoint,model,optimizer,step,manifest,states)
                # The dedicated evaluation card was checked at launch. Do not
                # terminate other jobs if its availability changes.
                eval_step=step;eval_path=out/f'development_step_{step:06d}';log=(out/f'evaluation_{step:06d}.log').open('ab',buffering=0)
                evaluation=subprocess.Popen([sys.executable,'-u','evaluate_rollout_proof.py','--checkpoint',str(checkpoint),'--out',str(eval_path)],
                    cwd=Path(__file__).parent,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',RANK='0',LOCAL_RANK='0',WORLD_SIZE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic(out/'evaluation_launcher.json',dict(pid=evaluation.pid,update=step,path=str(eval_path),time=time.time()))
            if world>1:dist.barrier()
    assert all(torch.equal(value,model.state_dict()[name].detach().cpu()) for name,value in frozen_state.items())
    if rank==0:
        atomic(out/'frozen_representation_verified.json',dict(passed=True,parameter_tensors=len(frozen_state),update=step,time=time.time()))
        collect_evaluation(wait=True)
        atomic(out/'status.json',dict(state='training_budget_complete',pid=os.getpid(),gpus=visible,update=step,time=time.time(),goal_complete=False))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        if int(os.environ.get('RANK',0))==0 and '--out' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--out')+1])
            if folder.exists():atomic(folder/'status.json',dict(state='failed',pid=os.getpid(),time=time.time(),error=repr(error),traceback=traceback.format_exc(),goal_complete=False))
        raise
