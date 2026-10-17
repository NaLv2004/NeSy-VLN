"""Joint evidence/rule fitting on normal near/medium measured trajectories."""
import argparse,json,os,time,gc
import torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import FactoredWorld,Config
from data import ROOT,sha
from joint_data import JointData
from joint_objective import JointObjective,configure,TRAINABLE
from accelerated_transition import install
from autobatch import choose
from evaluate import atomic,evaluate,summarize

OUT=ROOT/'run_stable_rules_v75'
def log(name,row):
    with (OUT/name).open('a') as f:f.write(json.dumps(row)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume');args=parser.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2,3' and int(os.environ['WORLD_SIZE'])==2
    torch.cuda.set_device(local);torch.set_num_threads(3);torch.manual_seed(671017)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dist.init_process_group('nccl');device=torch.device('cuda',local)
    old=ROOT/'parallel_prefix_v75';commit=json.loads((old/'source.COMMITTED.json').read_text());assert sha(old/'source.pt')==commit['sha256']
    saved=torch.load(old/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**dict(saved['model_config'],linear_visual_evidence=True))).to(device)
    missing,unexpected=model.load_state_dict(saved['model'],strict=False)
    expected=set() if saved['model_config'].get('linear_visual_evidence',False) else {'linear_visual_evidence.weight','evidence_mean','evidence_scale'}
    assert not unexpected and set(missing)==expected
    install(model);model.eval().requires_grad_(False)
    data=JointData(model,saved['manifest']['transition_holdout_episodes'],device)
    # A zero adapter must reproduce the original program's local observations.
    frames=data.feedback[:512,0];programs=data.feedback[:512,1]
    with torch.no_grad():
        observed,base=data.logits(model,frames,programs)
        if expected:assert torch.equal(observed,base)
        native=model.observe(data.x[frames],data.context[programs])
        assert torch.equal(observed.argmax(-1),native.argmax(-1))
    configure(model);objective=JointObjective(model,data)
    optimizer=torch.optim.AdamW([
        dict(params=list(model.linear_visual_evidence.parameters()),lr=1e-4,weight_decay=.1),
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('linear_visual_evidence.')],lr=5e-5,weight_decay=1e-4)])
    frozen={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(TRAINABLE) and n not in ('evidence_mean','evidence_scale')}
    manifest=dict(saved['manifest']);del saved
    manifest.update(architecture='stable_rules_v75',source_commit=commit,curriculum=data.protocol,data_selection=data.summary,
        updates=1000,learning_rates=dict(visual_linear=1e-4,rules=5e-5),trainable_prefixes=TRAINABLE,
        frozen_context_definitions_goal_and_memory_predictor=True,visual_evidence_frozen=True,anonymous_goal_ids=True,no_new_action_head=True,no_runtime_feedback=True,
        training_root_memory='Recorded symbolic rollout memory, matching deployed per-step carrier instead of re-grounding from every image',
        objective='v68 evidence/dynamics/calibration objective on768fresh normalnear/medium training-layout episodes; only observation-transition decoder trains; source44 visual and goal symbols remain fixed',
        max_actions=32,minimum_actions_between_rule_renewal=5,finalfixed80_used=False,
        full_original_dev80_replaced_by_user=True,development80_sha256=data.protocol['development_sha256'],
        eligibility='normal near/medium only,pre-frozen fitting/heldout measured episodes; entirely distinct from old200episodes; dev layouts disjoint',
        checkpoint_selection='Minimum heldout goal NLL + O transition NLL +4*symbolic goal Brier, with no lower goal TPR and at most2points lower one/two-step accuracy',
        inference='Same float64 discrete-rule Bellman derivation batched on GPU; memory grounded only on rule renewal',gpu_symbolic_planner=True,validation_workers_per_gpu=40,
        source_reference='https://proceedings.neurips.cc/paper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf',
        reference_scope='Planning-relevant model fitting motivation; exact finite-sequence categorical goal calibration is our adaptation, not a reproduction')
    if rank==0:atomic(OUT/'manifest.json',manifest);atomic(OUT/'dataset.json',data.summary)
    rng=torch.Generator(device=device).manual_seed(671017+rank)
    first=1;best=float('inf');chosen=None;feedback_size=4096;pair_size=512
    if args.resume:
        p=OUT/args.resume;c=json.loads(p.with_suffix('.COMMITTED.json').read_text());assert sha(p)==c['sha256']
        r=torch.load(p,map_location=device,weights_only=False);model.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer'])
        first=r['update']+1;best=r['best'];chosen=r['chosen'];feedback_size=r['feedback_size'];pair_size=r['pair_size']
        rng.set_state(r['rng_states'][rank].cpu());del r

    if first==1:
        model.train();selected_batch,capacity=choose(model,objective,data,rng)
        feedback_size=selected_batch['feedback'];pair_size=selected_batch['transitions']
        if rank==0:atomic(OUT/'capacity.json',capacity)

    wrapper=DDP(objective,device_ids=[local],broadcast_buffers=False)

    @torch.no_grad()
    def score(update):
        model.eval();d=data.parts['holdout'];totals=torch.zeros(15,device=device,dtype=torch.float64)
        for ids in d['feedback'][rank::2].split(2048):
            _,m=objective.feedback(data.feedback[ids])
            totals[:8]+=torch.stack([m[k].double() for k in ('positive','negative','tp','fp','positive_nll','negative_nll','anchored','code_preserved')])
        # Fixed deterministic subset of training-layout holdout transitions, not dev episodes.
        indices=d['pairs'][::2][rank::2]
        for ids in indices.split(256):
            _,m=objective.transitions(data.pairs[ids]);n=float(m['one_count']);n2=float(m['two_count'])
            totals[8:13]+=torch.tensor([float(m['one_hits']),n,float(m['two_hits']),n2,
                float(m['first_o_nll'])*n+float(m['open_o_nll'])*n2],device=device,dtype=torch.float64)
            calibrated,_=objective.calibrated_goal(data.pairs[ids]);totals[13]+=calibrated.double()*n;totals[14]+=n
        dist.all_reduce(totals);p,n,tp,fp,pl,nl,anchor,preserved,h1,c1,h2,c2,tnll,cb,cn=totals.cpu().tolist()
        row=dict(update=update,time=time.time(),positive=int(p),negative=int(n),tpr=tp/p,fpr=fp/n,
            goal_nll=.5*(pl/p+nl/n),negative_code_agreement=preserved/max(1,anchor),
            one_accuracy=h1/c1,two_accuracy=h2/c2,transition_nll=tnll/(c1+c2),symbolic_goal_brier=cb/cn)
        row['selection_score']=row['goal_nll']+row['transition_nll']+4*row['symbolic_goal_brier']
        if rank==0:log('heldout.jsonl',row)
        model.train();return row

    if first==1:
        baseline=score(0)
        if rank==0:atomic(OUT/'baseline_holdout.json',baseline)
    else:baseline=json.loads((OUT/'baseline_holdout.json').read_text())
    def save(update):
        states=[None,None];dist.all_gather_object(states,rng.get_state())
        if rank==0:
            p=OUT/f'checkpoint_{update:06d}.pt';tmp=p.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),model_config=model.export_config(),manifest=manifest,update=update,
                optimizer=optimizer.state_dict(),rng_states=states,best=best,chosen=chosen,feedback_size=feedback_size,pair_size=pair_size),tmp)
            tmp.replace(p);c=dict(filename=p.name,update=update,sha256=sha(p),time=time.time())
            atomic(p.with_suffix('.COMMITTED.json'),c);atomic(OUT/'latest.COMMITTED.json',c)
            if chosen==update:atomic(OUT/'best.COMMITTED.json',c)
            keep={p.name}
            if (OUT/'best.COMMITTED.json').exists():keep.add(json.loads((OUT/'best.COMMITTED.json').read_text())['filename'])
            for q in OUT.glob('checkpoint_*.pt'):
                if q.name not in keep and q.with_suffix('.COMMITTED.json').exists():q.unlink();q.with_suffix('.COMMITTED.json').unlink()
    began=time.time()
    for update in range(first,1001):
        tick=time.time();model.train();f,p=data.sample(rng,feedback_size,pair_size)
        optimizer.zero_grad(set_to_none=True);loss,metrics=wrapper(f,p);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],5.)
        assert torch.isfinite(loss) and torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
        row=dict(update=update,time=time.time(),seconds=time.time()-tick,feedback_batch=feedback_size,transition_batch=pair_size,
            gradient_norm=float(norm),peak_gib=torch.cuda.max_memory_allocated()/2**30,**{k:float(v) for k,v in metrics.items()})
        log(f'rank{rank}.jsonl',row);atomic(OUT/f'status_rank{rank}.json',dict(state='training',pid=os.getpid(),**row))
        if False: # Capacity was measured on the complete dataset before training.
            peak=torch.tensor(row['peak_gib'],device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
            if peak<7:feedback_size=8192;pair_size=1024
            if rank==0:atomic(OUT/'capacity.json',dict(measured_peak_gib=float(peak),feedback_batch=feedback_size,transition_batch=pair_size,measured_real_backward=True))
        if update%100==0:
            measured=score(update)
            qualifies=(measured['tpr']>=baseline['tpr'] and measured['fpr']<=max(.02,baseline['fpr']) and
                measured['one_accuracy']>=baseline['one_accuracy']-.02 and measured['two_accuracy']>=baseline['two_accuracy']-.02 and
                measured['symbolic_goal_brier']<baseline['symbolic_goal_brier'] and measured['selection_score']<baseline['selection_score'])
            if qualifies and measured['selection_score']<best:best=measured['selection_score'];chosen=update
            for n,v in frozen.items():assert torch.equal(v,model.state_dict()[n].detach().cpu()),n
            save(update);dist.barrier()
            if rank==0:print(json.dumps(dict(update=update,heldout=measured,qualifies=qualifies,elapsed=time.time()-began)),flush=True)
    summary=None;selected=None
    if chosen is not None:
        selected=json.loads((OUT/'best.COMMITTED.json').read_text());p=OUT/selected['filename'];assert sha(p)==selected['sha256']
        model.load_state_dict(torch.load(p,map_location=device,weights_only=False)['model']);model.detach_memory_carrier=True;model.eval()
        del wrapper,objective,optimizer,data,loss,metrics;gc.collect();torch.cuda.empty_cache()
        folder=OUT/'development_best'
        if not (folder/'summary.json').exists():
            atomic(OUT/f'status_rank{rank}.json',dict(state='evaluating_new80',pid=os.getpid(),update=chosen,time=time.time()))
            rows=evaluate(model,manifest,folder,rank,2,workers=40);all_rows=[None,None];dist.all_gather_object(all_rows,rows)
            if rank==0:atomic(folder/'summary.json',summarize([r for rows in all_rows for r in rows]))
        dist.barrier();summary=json.loads((folder/'summary.json').read_text())
    atomic(OUT/f'status_rank{rank}.json',dict(state='complete',pid=os.getpid(),time=time.time(),updates=1000))
    if rank==0:atomic(OUT/'result.json',dict(state='complete',updates=1000,time=time.time(),chosen=selected,development=summary,
        baseline_development=json.loads((ROOT/'run_goal_probability_v68/development_best/summary.json').read_text()),goal_complete=False,finalfixed80_used=False))
    dist.destroy_process_group()
if __name__=='__main__':main()
