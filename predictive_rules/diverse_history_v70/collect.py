"""Fresh normal short/medium on-policy observations and measured alternatives."""
import argparse,fcntl,json,os,select,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from data import ROOT,sha
from model import FactoredWorld,Config
from accelerated_transition import install
from curriculum import make_bank
from policy import Policy
from batch_future import BatchFutureEncoder
from predictive_nesy.environment import FrozenVision
from predictive_nesy.gate import NeuralBoundaryGate
OUT=ROOT/'diverse_history_v70';SOURCE=Path(__file__).resolve().parent
def atomic(p,row):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)

class Worker:
    def __init__(self,bank,rank,i):
        sys.path.insert(0,'/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')
        from objectnav_rl.ipc import send,receive
        self.send,self.receive=send,receive;self.log=(OUT/f'sim_{rank}_{i}.log').open('ab')
        self.p=subprocess.Popen([bank.env['sim_python'],str(SOURCE/'worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        self.call('init',experiment=bank.env,train_scenes=bank.provenance['train_layouts'])
    def call(self,op,**args):
        self.send(self.p.stdin,dict(op=op,**args))
        if not select.select([self.p.stdout],[],[],180)[0]:raise TimeoutError((self.p.pid,op))
        r=self.receive(self.p.stdout)
        if not r['ok']:raise RuntimeError(r['error'])
        return r['result']
    def close(self):
        if self.p.poll() is None:self.call('close');self.p.wait(timeout=20)
        self.log.close()

class CachedEncoder:
    def __init__(self):self.raw=None
    def encode_joint(self,images):assert len(images)==1 and self.raw is not None;return None,self.raw

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);rank=p.parse_args().rank
    assert rank in (0,1) and os.environ['CUDA_VISIBLE_DEVICES']==str(6+rank)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    lock=(OUT/f'rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=json.loads((OUT/'manifest.json').read_text());commit=json.loads((OUT/'source.COMMITTED.json').read_text());assert sha(OUT/'source.pt')==commit['sha256']
    saved=torch.load(OUT/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);install(model)
    frontend=FrozenVision(saved['manifest']['frozen_vision']);del saved
    gate=NeuralBoundaryGate([model,frontend.vision],interval=1);encoder=BatchFutureEncoder(frontend,gate,512);cached=CachedEncoder()
    active_policy=[None]
    def guard(*_):assert active_policy[0] is not None and active_policy[0].grounding_allowed,'Actual memory correction outside renewal'
    mh=model.memory.register_forward_pre_hook(guard)
    ih=model.inverse.register_forward_pre_hook(lambda *_:(_ for _ in ()).throw(RuntimeError('Inverse action selection forbidden')))
    bank=make_bank();assigned=info['episodes'][rank::2];holdout=set(info['holdout_ids']);folder=OUT/'episodes'
    assert not {e['scene'] for e in assigned}&{e['scene'] for e in bank.dev}
    pending=[];done=frames=outcomes=successes=0
    for e in assigned:
        path=folder/(e['episode_id']+'.npz')
        if path.exists() and path.with_suffix('.json').exists():
            meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['sha256']
            done+=1;frames+=meta['frames'];outcomes+=meta['outcomes'];successes+=int(meta['success'])
        else:pending.append(e)
    began=time.time();sims=[];status=dict(state='starting',pid=os.getpid(),rank=rank,total=len(assigned),completed=done,
        frames=frames,outcomes=outcomes,successes=successes,workers=24,feature_batch=512,time=began)
    atomic(OUT/f'status_rank{rank}.json',status)
    try:
        sims=[Worker(bank,rank,i) for i in range(24)]
        with ThreadPoolExecutor(max_workers=24) as pool:
            for start in range(0,len(pending),24):
                batch=pending[start:start+24];responses=list(pool.map(lambda z:z[0].call('reset',episode=z[1]),zip(sims,batch)))
                rgbs=[r['rgb'] for r in responses];labels=[[r['eligible']] for r in responses]
                policies=[Policy(model,e['category'],cached,gate) for e in batch]
                buffers=[defaultdict(list) for _ in batch];active=list(range(len(batch)));success=[False]*len(batch)
                last_raw={};failed=set()
                for step in range(32):
                    with gate.boundary(step):raw=encoder.encode([Image.fromarray(rgbs[i]) for i in active])
                    prepared={}
                    for j,i in enumerate(active):
                        cached.raw=raw[j:j+1];active_policy[0]=policies[i]
                        prepared[i]=policies[i].choose(rgbs[i],defer_planning=True);active_policy[0]=None
                        buffers[i]['features'].append(raw[j].cpu().numpy());last_raw[i]=False
                    choices=dict(pool.map(lambda i:(i,policies[i].complete_choice(*prepared[i])),active))
                    requests=[]
                    for i in active:
                        action,_=choices[i]
                        if action is None:failed.add(i)
                        requests.append((i,action,pool.submit(sims[i].call,'step',action=action,probe=step%4==0 or action in (0,None),
                            seed=701017+step*97+start*109+i*37+rank*100000)))
                    responses=[future.result() for _,_,future in requests]
                    images=[];bounds=[0]
                    for response in responses:
                        images.extend(Image.fromarray(x) for x in response['future_images']);bounds.append(len(images))
                    with gate.boundary(step):future_raw=encoder.encode(images).cpu().numpy() if images else None
                    next_active=[]
                    for j,((i,action,_),response) in enumerate(zip(requests,responses)):
                        b=buffers[i];n=bounds[j+1]-bounds[j]
                        if n:
                            b['future_features'].extend(future_raw[bounds[j]:bounds[j+1]])
                            b['future_stop_success'].extend(response['future_labels']);b['root_step'].extend([step]*n)
                            b['first_action'].extend(response['first_action']);b['second_action'].extend(response['second_action'])
                        if action is not None:
                            policies[i].commit_execution(action);labels[i].append(response['eligible']);rgbs[i]=response['actual'];last_raw[i]=True
                            success[i]=response['success']
                            if not response['done']:next_active.append(i)
                    active=next_active
                    if not active:break
                terminal=[i for i in range(len(batch)) if last_raw.get(i,False)]
                if terminal:
                    with gate.boundary(32):end_features=encoder.encode([Image.fromarray(rgbs[i]) for i in terminal]).cpu().numpy()
                    for j,i in enumerate(terminal):buffers[i]['features'].append(end_features[j])
                for i,e in enumerate(batch):
                    b=buffers[i];policy=policies[i];arrays={
                        'features':np.asarray(b['features'],np.float16),'actions':np.asarray(policy.actions,np.int16),
                        'stop_success':np.asarray(labels[i],np.bool_),'boundaries':np.asarray([r['start'] for r in policy.programs],np.int16),
                        'future_features':np.asarray(b['future_features'],np.float16).reshape(-1,49,128),
                        'future_stop_success':np.asarray(b['future_stop_success'],np.bool_),
                        **{key:np.asarray(b[key],np.int16) for key in ('root_step','first_action','second_action')}}
                    assert len(arrays['features'])==len(policy.actions)+1==len(labels[i])
                    path=folder/(e['episode_id']+'.npz');tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
                    meta=dict(episode=e['episode_id'],scene=e['scene'],category=e['category'],distance_group=e['distance_group'],domain='normal',
                        part='holdout' if e['episode_id'] in holdout else 'fit',training_layout=True,success=success[i],planning_failed=i in failed,
                        frames=len(labels[i]),outcomes=len(arrays['future_features']),actions=len(policy.actions),
                        source_sha256=commit['sha256'],sha256=sha(path),programs=policy.programs,steps=policy.steps,
                        no_future_observation_in_policy=True,all_executed_actions_symbolic=True)
                    atomic(path.with_suffix('.json'),meta);done+=1;frames+=meta['frames'];outcomes+=meta['outcomes'];successes+=int(meta['success'])
                status.update(state='collecting',completed=done,frames=frames,outcomes=outcomes,successes=successes,time=time.time(),
                    seconds=time.time()-began,peak_torch_gib=torch.cuda.max_memory_allocated()/2**30)
                atomic(OUT/f'status_rank{rank}.json',status);print(json.dumps(status),flush=True)
        atomic(OUT/f'status_rank{rank}.json',dict(status,state='complete',time=time.time()))
    except Exception as error:
        atomic(OUT/f'status_rank{rank}.json',dict(status,state='failed',time=time.time(),error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        ih.remove();mh.remove();gate.close()
        for sim in sims:sim.close()

if __name__=='__main__':main()
