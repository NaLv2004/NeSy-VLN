"""Distributed discrete-symbol dynamics training; not a navigation controller."""
import argparse,fcntl,json,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from predictive_nesy.discrete_visual_world import DiscreteVisualWorld,WorldConfig
from predictive_nesy.data import sha

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)
def save(path,model,opt,step,manifest):
    tmp=path.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),model_config=model.export_config(),optimizer=opt.state_dict(),update=step,manifest=manifest),tmp)
    os.replace(tmp,path);atomic(path.with_suffix('.COMMITTED.json'),dict(path=path.name,sha256=sha(path),update=step,time=time.time()))

@torch.no_grad()
def evaluate(model,data,ids):
    model.eval();correct=0;total=0;changed_correct=0;changed_total=0;exact=0;loss=0.;copied=0;predicted_changes=0;by_action={}
    for start in range(0,len(ids),1024):
        ix=ids[start:start+1024];before=data['before'][ix].long();target=data['after'][ix].long();action=data['actions'][ix]
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(before,action)
        prediction=logits.argmax(-1);match=prediction==target;changed=target!=before
        correct+=int(match.sum());total+=match.numel();changed_correct+=int((match&changed).sum());changed_total+=int(changed.sum())
        exact+=int(match.all(-1).sum());copied+=int((before==target).sum());predicted_changes+=int((prediction!=before).sum())
        loss+=float(F.cross_entropy(logits.float().flatten(0,1),target.flatten(),reduction='sum'))
        for a in range(17):
            mask=action==a;r=by_action.setdefault(a,dict(correct=0,total=0,changed_correct=0,changed_total=0))
            r['correct']+=int(match[mask].sum());r['total']+=match[mask].numel();r['changed_correct']+=int((match[mask]&changed[mask]).sum());r['changed_total']+=int(changed[mask].sum())
    return dict(transitions=len(ids),patch_accuracy=correct/total,changed_patch_accuracy=changed_correct/max(changed_total,1),
        whole_frame_exact=exact/len(ids),copy_baseline=copied/total,ce=loss/total,
        actual_changed_fraction=changed_total/total,predicted_changed_fraction=predicted_changes/total,by_action=by_action)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--updates',type=int,default=2000);p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--eval-every',type=int,default=200);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert not set(visible)-{'3','4','5','7'} and len(visible)==world
    torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(3515)
    if world>1:dist.init_process_group('nccl')
    folder=ROOT/'discrete_visual_world_data_v15';dm=json.loads((folder/'manifest.json').read_text());assert sha(folder/'transitions.npz')==dm['npz_sha256']
    with np.load(folder/'transitions.npz',allow_pickle=False) as source:
        data={k:torch.tensor(source[k],device='cuda') for k in ['before','after','actions']};held=source['held'];kind=source['kind']
    selections={k:torch.tensor(np.flatnonzero((~held)&(kind==k)),device='cuda') for k in [0,1]};validation=torch.tensor(np.flatnonzero(held),device='cuda')
    model=DiscreteVisualWorld().cuda();wrapper=DDP(model,device_ids=[local]) if world>1 else model
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    out=Path(a.out).resolve();assert out.parent==ROOT
    if rank==0:
        out.mkdir(exist_ok=True);lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'latest.pt').exists():raise RuntimeError('Existing run must not restart from scratch')
    manifest=dict(architecture='discrete_visual_symbol_world_v15',model=model.export_config(),data=dm,
        input=['49 learned visual symbol IDs','action ID'],output='49 categorical distributions over64 learned symbols',
        no_continuous_state_target=True,no_semantic_state_labels=True,no_navigation_controller=True,
        train_args=vars(a),world_size=world,trainable_parameters=sum(p.numel() for p in model.parameters()),created=time.time())
    if rank==0:atomic(out/'manifest.json',manifest)
    generator=torch.Generator(device='cuda').manual_seed(3515+rank);began=time.time();best=-1
    total=1 if a.preflight else a.updates
    for step in range(1,total+1):
        model.train();begin=time.time();n=a.batch_size//2
        ix=torch.cat([selections[k][torch.randint(len(selections[k]),(n,),device='cuda',generator=generator)] for k in [0,1]])
        before=data['before'][ix].long();target=data['after'][ix].long();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=wrapper(before,data['actions'][ix]);loss=F.cross_entropy(logits.flatten(0,1),target.flatten())
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();torch.cuda.synchronize()
        with torch.no_grad():
            prediction=logits.argmax(-1);changed=target!=before
            metrics=torch.tensor([float(loss),float((prediction==target).float().mean()),float(((prediction==target)&changed).sum()/changed.sum().clamp_min(1))],device='cuda')
            if world>1:dist.all_reduce(metrics);metrics/=world
        row=dict(update=step,time=time.time(),seconds=time.time()-begin,elapsed_seconds=time.time()-began,
            loss=float(metrics[0]),patch_accuracy=float(metrics[1]),changed_patch_accuracy=float(metrics[2]),
            memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,global_batch=a.batch_size*world)
        if rank==0:
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            atomic(out/'status.json',dict(state='training',pid=os.getpid(),gpus=visible,**row))
            if step==1 or step%100==0:print(json.dumps(row),flush=True)
        if a.preflight:
            if rank==0:atomic(out/'preflight.json',dict(passed=True,**row))
            break
        if step==1 or step%100==0:
            if rank==0:save(out/'latest.pt',model,opt,step,manifest)
        if step%a.eval_every==0 or step==total:
            if world>1:dist.barrier()
            if rank==0:
                result=dict(update=step,training_only=True,split='internal_episode_holdout',**evaluate(model,data,validation))
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                save(out/f'checkpoint_{step:06d}.pt',model,opt,step,manifest)
                if result['changed_patch_accuracy']>best:
                    best=result['changed_patch_accuracy'];atomic(out/'best_world_validation.json',result)
                print(json.dumps({k:v for k,v in result.items() if k!='by_action'}),flush=True)
            if world>1:dist.barrier()
    if rank==0:atomic(out/'status.json',dict(state='preflight_complete' if a.preflight else 'training_budget_complete',update=total,pid=os.getpid(),time=time.time(),goal_complete=False,no_navigation_result=True))
    if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
