"""Bounded multi-step dynamics fit; no navigation model is promoted here."""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from predictive_nesy.discrete_visual_world import WorldConfig
from predictive_nesy.discrete_rollout_world import DiscreteRolloutWorld
from predictive_nesy.subgoal_data import load_windows
from predictive_nesy.data import sha
from train_discrete_visual_world_v15 import atomic,save,evaluate

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')

@torch.no_grad()
def audit_rollout(model,windows,selection):
    model.eval();counts=torch.zeros((10,6),device='cuda',dtype=torch.float64)
    for begin in range(0,len(selection),64):
        ix=selection[begin:begin+64];target=windows['targets'][ix];valid=windows['action_valid'][ix]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=model.rollout(target[:,0],windows['actions'][ix]).argmax(-1)
        target=target[:,1:];match=prediction==target;changed=target!=windows['targets'][ix,:-1]
        mask=valid[:,:,None];changed=changed&mask
        counts[:,0]+=(match&mask).sum((0,2));counts[:,1]+=mask.sum((0,2))*49
        counts[:,2]+=(match&changed).sum((0,2));counts[:,3]+=changed.sum((0,2))
        counts[:,4]+=(match.all(-1)&valid).sum(0);counts[:,5]+=valid.sum(0)
    rows=[]
    for t,row in enumerate(counts.cpu().tolist(),1):
        correct,total,cc,ct,exact,n=row
        rows.append(dict(step=t,windows=int(n),patch_accuracy=correct/max(total,1),
                         changed_patch_accuracy=cc/max(ct,1),whole_frame_exact=exact/max(n,1)))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--preflight',action='store_true')
    p.add_argument('--updates',type=int,default=600);p.add_argument('--sequence-batch',type=int,default=128)
    p.add_argument('--branch-batch',type=int,default=1024);p.add_argument('--eval-every',type=int,default=200);a=p.parse_args()
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert set(visible)<=set(['3','4','5','7']) and len(visible)==world
    torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(3616)
    if world>1:dist.init_process_group('nccl')
    out=Path(a.out).resolve();assert out.parent==ROOT
    if rank==0:
        out.mkdir(exist_ok=True);lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'latest.pt').exists():raise RuntimeError('Refuse to overwrite an existing checkpoint')
    source=ROOT/'run_discrete_visual_world_v15/checkpoint_000800.pt'
    assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False);dm=saved['manifest']['data']
    model=DiscreteRolloutWorld(WorldConfig(**saved['model_config'])).cuda();model.load_state_dict(saved['model'])
    folder=ROOT/'discrete_visual_world_data_v15';assert sha(folder/'transitions.npz')==dm['npz_sha256']
    with np.load(folder/'transitions.npz',allow_pickle=False) as data:
        pairs={k:torch.tensor(data[k],device='cuda') for k in ['before','after','actions']}
        fit=torch.tensor(np.flatnonzero(~data['held']),device='cuda');held=torch.tensor(np.flatnonzero(data['held']),device='cuda')
    rows,provenance=load_windows(dm['data']['source'],dm['data']['train_layouts'])
    excluded=np.asarray([int(hashlib.sha256(w['episode_id'].encode()).hexdigest()[:8],16)%5==0 for w in rows])
    windows={k:torch.tensor(np.asarray([w[k] for w in rows]),device='cuda') for k in ['targets','actions','action_valid']}
    wfit=torch.tensor(np.flatnonzero(~excluded),device='cuda');wheld=torch.tensor(np.flatnonzero(excluded),device='cuda')
    assert not(set(w['episode_id'] for w,h in zip(rows,excluded) if h)&set(w['episode_id'] for w,h in zip(rows,excluded) if not h))
    manifest=dict(architecture='discrete_visual_symbol_autonomous_rollout_v16',source=str(source),source_sha256=sha(source),data=dm,
        demonstration_provenance=provenance,train_args=vars(a),trainable_parameters=sum(x.numel() for x in model.parameters()),
        world_size=world,component_only=True,no_navigation_result=True,no_manual_semantic_states=True,
        rollout_input=['initial visual symbol IDs','known action IDs'],future_observations='labels_only',
        forward_states='hard categorical IDs; straight-through derivatives in training only',
        losses={'one_step_discrete_cross_entropy':1.,'autonomous_10_step_discrete_cross_entropy':1.},
        optimizer='Fresh AdamW lr1e-4; initialized from the complete v15 component checkpoint',time=time.time())
    if rank==0:atomic(out/'manifest.json',manifest)
    # Check exact categorical execution agreement before any parameter update.
    with torch.no_grad():
        initial=windows['targets'][wheld[:16],0];actions=windows['actions'][wheld[:16]]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.rollout(initial,actions).argmax(-1);current=initial
            direct=[]
            for action in actions.unbind(1):current=model(current,action).argmax(-1);direct.append(current)
        exact=bool(torch.equal(output,torch.stack(direct,1)))
        if not exact:raise RuntimeError('Soft surrogate changed the discrete forward rollout')
    if rank==0:
        atomic(out/'categorical_forward_contract.json',dict(passed=exact,neural_policy_deployed=False,future_frames_input=False))
        baseline=dict(update=0,training_only=True,rollout=audit_rollout(model,windows,wheld))
        atomic(out/'baseline_rollout.json',baseline)
    if world>1:dist.barrier()
    wrapper=DDP(model,device_ids=[local]) if world>1 else model
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    generator=torch.Generator(device='cuda').manual_seed(3616+rank);started=time.time();total=1 if a.preflight else a.updates
    for step in range(1,total+1):
        model.train();began=time.time();optimizer.zero_grad(set_to_none=True)
        ix=fit[torch.randint(len(fit),(a.branch_batch,),generator=generator,device='cuda')]
        wx=wfit[torch.randint(len(wfit),(a.sequence_batch,),generator=generator,device='cuda')]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            single,sequence=wrapper(pairs['before'][ix].long(),pairs['actions'][ix],windows['targets'][wx,0],windows['actions'][wx])
            single_loss=F.cross_entropy(single.flatten(0,1),pairs['after'][ix].long().flatten())
            token_loss=F.cross_entropy(sequence.flatten(0,2),windows['targets'][wx,1:].flatten(),reduction='none').reshape(len(wx),10,49)
            mask=windows['action_valid'][wx].float();rollout_loss=(token_loss.mean(-1)*mask).sum()/mask.sum().clamp_min(1)
            loss=single_loss+rollout_loss
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
        metrics=torch.tensor([float(loss.detach()),float(single_loss.detach()),float(rollout_loss.detach())],device='cuda')
        if world>1:dist.all_reduce(metrics);metrics/=world
        row=dict(update=step,time=time.time(),seconds=time.time()-began,elapsed_seconds=time.time()-started,
            loss=float(metrics[0]),one_step_loss=float(metrics[1]),rollout_loss=float(metrics[2]),
            memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,global_sequence_batch=a.sequence_batch*world,
            global_pair_batch=a.branch_batch*world)
        if rank==0:
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            atomic(out/'status.json',dict(state='training',pid=os.getpid(),**row))
            if step==1 or step%50==0:print(json.dumps(row),flush=True)
        if a.preflight:
            if rank==0:atomic(out/'preflight.json',dict(passed=True,**row))
            break
        if step==1 or step%100==0:
            if rank==0:save(out/'latest.pt',model,optimizer,step,manifest)
        if step%a.eval_every==0 or step==total:
            if world>1:dist.barrier()
            if rank==0:
                result=dict(update=step,training_only=True,split='internal_episode_holdout',
                    one_step=evaluate(model,pairs,held),rollout=audit_rollout(model,windows,wheld))
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                save(out/f'checkpoint_{step:06d}.pt',model,optimizer,step,manifest)
                print(json.dumps(dict(event='evaluation',update=step,rollout=result['rollout'])),flush=True)
            if world>1:dist.barrier()
    if rank==0:atomic(out/'status.json',dict(state='preflight_complete' if a.preflight else 'training_budget_complete',update=total,
        time=time.time(),goal_complete=False,no_navigation_result=True))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
