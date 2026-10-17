"""Joint discrete representation/rule fitting with factual goal feedback."""
import json,os,random,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import FactoredWorld,Config
from data import Corpus,ROOT,sha
from evaluate import atomic,evaluate,summarize
from feedback_objective import FeedbackData,FeedbackObjective,feedback
from accelerated_transition import install

OUT=ROOT/'run_goal_discrimination_v64';DATA=ROOT/'goal_discrimination_v64'
def log(name,row):
    with (OUT/name).open('a') as f:f.write(json.dumps(row)+'\n')

def exclude_feedback_holdout(corpus,names):
    corpus.names=[n for n in corpus.names if n not in names]
    corpus.terminal_pools={task:[ids[-1] for name,ids in corpus.episodes.items() if name not in names and corpus.rows[ids[0]]['category']==task]
        for task in ('chair','table')}
    def keep(rows):return [r for r in rows if corpus.branches[r[0]][0]['episode'] not in names]
    corpus.old_branch_positions=keep(corpus.old_branch_positions)
    corpus.onpolicy_positions=keep(corpus.onpolicy_positions)
    corpus.branch_positions=keep(corpus.branch_positions)
    for label,rows in (('original',corpus.old_branch_positions),('onpolicy',corpus.onpolicy_positions)):
        corpus.phase_pools[label]=([r for r in rows if r[1]==0],[r for r in rows if 1<=r[1]<5],[r for r in rows if r[1]>=5])
        assert all(corpus.phase_pools[label])

def main():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6,7' and int(os.environ['WORLD_SIZE'])==2
    torch.cuda.set_device(local);torch.set_num_threads(3);torch.manual_seed(641017)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dist.init_process_group('nccl');device=torch.device('cuda',local);rng=random.Random(641017+rank)
    OUT.mkdir(exist_ok=True);assert not (OUT/'latest.COMMITTED.json').exists(),'Fresh experiment; inspect existing run instead of overwriting'
    commit=json.loads((DATA/'source.COMMITTED.json').read_text());assert sha(DATA/'source.pt')==commit['sha256']
    saved=torch.load(DATA/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).to(device);model.load_state_dict(saved['model']);install(model)
    measured=FeedbackData(DATA,saved['manifest']['transition_holdout_episodes'],device)
    corpus=Corpus(rank,2,str(ROOT/'onpolicy_successors_v60'));exclude_feedback_holdout(corpus,measured.holdout)
    manifest=dict(saved['manifest']);del saved
    representation=('visual.','observation_definitions','memory.','memory_definitions')
    optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(representation)],lr=2e-5),
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(representation)],lr=1e-4)],weight_decay=1e-4)
    wrapper=DDP(FeedbackObjective(model),device_ids=[local],broadcast_buffers=False)
    # Larger measured-feedback batches reuse GPU-resident features. The main
    # trajectory batch preserves the known stable joint-fitting configuration.
    sizing=json.loads((DATA/'training_capacity.json').read_text())
    assert sizing['verified_finite_backward'] and sizing['peak_limit_gib']==20
    batch_demo,batch_branch,batch_feedback=sizing['selected_batch'];updates=1000
    manifest.update(architecture='goal_discrimination_v64',source_commit=commit,
        updates=updates,batch_demo_per_gpu=batch_demo,batch_branch_per_gpu=batch_branch,batch_feedback_per_gpu=batch_feedback,
        feedback_weight=1.0,feedback_fit_episodes=[n for n in measured.names if n not in measured.holdout],
        feedback_holdout_episodes=sorted(measured.holdout),
        feedback='Existing binary STOP outcome supervises equality to the model-generated goal observation code; no fixed semantic ID',
        feedback_sampling='Equal positive/negative decision positions from fitting episodes',
        abstraction_and_goal_frozen=False,learning_rates=dict(representation=2e-5,other=1e-4),
        no_new_runtime_head=True,no_runtime_reward_table=True,no_feedback_input_at_runtime=True,
        added_feedback_holdout_is_not_unseen_layout_validation=True,finalfixed80_used=False,
        original_full_dev80=True,runtime_actions='Unchanged NumPy categorical probability derivation only',
        feedback_reference='https://proceedings.mlr.press/v162/hansen-estruch22a.html',
        reference_scope='Goal-discriminating representation motivation only, not a GCB reproduction or bisimulation proof',
        efficiency='Pruned unused legacy O decoder; measured feedback features resident on GPU',
        capacity_benchmark=sizing,batched_policy_vision=True,batched_encoding_rare_numeric_change_accepted=True,
        checkpoint_selection='Minimum held-out class-balanced goal-code log loss every100updates; one original dev80 if lower than baseline by5percent with TPR not reduced and FPR<=2percent')
    if rank==0:atomic(OUT/'manifest.json',manifest)

    @torch.no_grad()
    def heldout(update):
        model.eval();rows=measured.parts['holdout'][rank::2];totals=torch.zeros(6,device=device,dtype=torch.float64)
        for start in range(0,len(rows),128):
            _,m=feedback(model,measured.batch(rows[start:start+128]))
            totals+=torch.stack([m[k] for k in ('positive_count','negative_count','true_positive','false_positive','positive_nll','negative_nll')]).to(torch.float64)
        dist.all_reduce(totals);p,n,tp,fp,pl,nl=totals.cpu().tolist()
        row=dict(update=update,time=time.time(),positive=int(p),negative=int(n),tp=int(tp),fp=int(fp),
            true_positive_rate=tp/max(1,p),false_positive_rate=fp/max(1,n),
            balanced_nll=.5*(pl/max(1,p)+nl/max(1,n)))
        if rank==0:log('goal_holdout.jsonl',row)
        model.train();return row

    def save(update):
        states=[None,None];dist.all_gather_object(states,dict(python=rng.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
        if rank==0:
            name=f'checkpoint_{update:06d}.pt';p=OUT/name;tmp=p.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),model_config=model.export_config(),manifest=manifest,
                optimizer=optimizer.state_dict(),update=update,rank_rng_states=states),tmp);tmp.replace(p)
            c=dict(filename=name,update=update,sha256=sha(p),time=time.time())
            atomic(p.with_suffix('.COMMITTED.json'),c);atomic(OUT/'latest.COMMITTED.json',c);return c

    baseline=heldout(0);best=float('inf');selected=None;began=time.time()
    for update in range(1,updates+1):
        tick=time.time();model.train();base=corpus.sample(rng,batch_demo,batch_branch,device)
        base['world_updates_encoder']=update%5==0
        d=measured.sample(rng,batch_feedback);optimizer.zero_grad(set_to_none=True)
        loss,metrics=wrapper(base,d);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert torch.isfinite(loss) and torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
        row={k:float(v) for k,v in metrics.items()};row.update(update=update,time=time.time(),seconds=time.time()-tick,
            gradient_norm=float(norm),gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        log(f'rank{rank}.jsonl',row);atomic(OUT/f'status_rank{rank}.json',dict(state='training',pid=os.getpid(),update=update,time=time.time()))
        if update%100==0:
            held=heldout(update);c=save(update)
            qualifies=(held['true_positive_rate']>=baseline['true_positive_rate'] and held['false_positive_rate']<=.02)
            if qualifies and held['balanced_nll']<best:
                best=held['balanced_nll'];selected=held
                if rank==0:atomic(OUT/'best.COMMITTED.json',c)
            if rank==0:
                keep={c['filename']}
                if (OUT/'best.COMMITTED.json').exists():keep.add(json.loads((OUT/'best.COMMITTED.json').read_text())['filename'])
                for p in OUT.glob('checkpoint_*.pt'):
                    if p.name not in keep:
                        assert p.parent==OUT and p.with_suffix('.COMMITTED.json').exists();p.unlink();p.with_suffix('.COMMITTED.json').unlink()
                print(json.dumps(dict(update=update,heldout=held,elapsed=time.time()-began)),flush=True)
            dist.barrier()
    qualifies=selected is not None and best<baseline['balanced_nll']*.95
    summary=None;chosen=None
    if qualifies:
        chosen=json.loads((OUT/'best.COMMITTED.json').read_text());p=OUT/chosen['filename'];assert sha(p)==chosen['sha256']
        state=torch.load(p,map_location=device,weights_only=False);model.load_state_dict(state['model']);del state
        atomic(OUT/f'status_rank{rank}.json',dict(state='evaluating_full80',pid=os.getpid(),update=chosen['update'],time=time.time()))
        rows=evaluate(model,manifest,OUT/'dev_best_full80',rank,2,workers=12);all_rows=[None,None];dist.all_gather_object(all_rows,rows)
        if rank==0:summary=summarize([r for group in all_rows for r in group]);atomic(OUT/'dev_best_full80/summary.json',summary)
    dist.barrier()
    atomic(OUT/f'status_rank{rank}.json',dict(state='complete',pid=os.getpid(),updates=updates,time=time.time()))
    if rank==0:atomic(OUT/'result.json',dict(state='complete',time=time.time(),updates=updates,
        source=commit,baseline_goal_holdout=baseline,best_goal_holdout=selected,qualified_for_full80=qualifies,
        evaluated_checkpoint=chosen,development=summary,finalfixed80_used=False,goal_complete=False))
    dist.destroy_process_group()

if __name__=='__main__':main()
