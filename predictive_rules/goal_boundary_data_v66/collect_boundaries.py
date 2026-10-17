"""Training-only all-action visual neighborhoods around observed success boundaries."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_counterfactual_closure_v63'))
from data import sha
from evaluate import atomic
from batch_future import BatchFutureEncoder
from predictive_nesy.environment import FrozenVision,make_bank
from predictive_nesy.gate import NeuralBoundaryGate
from collect_closure import Probe

OUT=ROOT/'goal_boundary_data_v66';SOURCE=Path(__file__).resolve().parent

class Replay(Probe):
    def __init__(self,bank,folder):
        sys.path.insert(0,'/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')
        from objectnav_rl.ipc import send,receive
        self.send,self.receive=send,receive
        self.log=(folder/'simulator.log').open('ab')
        self.process=subprocess.Popen([bank.env['sim_python'],str(SOURCE/'boundary_worker.py')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        self.call('init',experiment=bank.env,train_scenes=bank.provenance['train_layouts'])

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);args=p.parse_args()
    assert args.rank in (0,1) and os.environ['CUDA_VISIBLE_DEVICES']==str(6+args.rank)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    OUT.mkdir(exist_ok=True);lock=(OUT/f'rank{args.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=ROOT/'goal_discrimination_v64';info=json.loads((ROOT/'counterfactual_closure_v63/manifest.json').read_text())
    bank=make_bank();by_id={e['episode_id']:e for e in bank.train}
    names=info['episode_ids'];assert len(names)==600 and all(n in by_id for n in names)
    assert not {by_id[n]['scene'] for n in names}&{e['scene'] for e in bank.dev}
    folder=OUT/'episodes';folder.mkdir(exist_ok=True);assigned=names[args.rank::2]
    pending=[];done=positions=positive=0
    for name in assigned:
        meta=folder/(name+'.json');data=meta.with_suffix('.npz')
        if meta.exists() and data.exists():
            r=json.loads(meta.read_text());assert sha(data)==r['sha256']
            done+=1;positions+=r['positions'];positive+=r['positive_stop_outcomes']
        else:pending.append(name)
    saved=torch.load(old/'source.pt',map_location='cpu',weights_only=False)
    frontend=FrozenVision(saved['manifest']['frozen_vision']);del saved
    gate=NeuralBoundaryGate([frontend.vision],interval=1);encoder=BatchFutureEncoder(frontend,gate,256)
    sims=[];began=time.time();state=dict(state='starting',pid=os.getpid(),completed=done,total=300,positions=positions,
        positive_stop_outcomes=positive,workers=24,time=began,batch_images=256)
    atomic(OUT/f'status_rank{args.rank}.json',state)
    try:
        for i in range(24):
            path=OUT/f'worker_{args.rank}_{i}';path.mkdir(exist_ok=True);sims.append(Replay(bank,path))
        with ThreadPoolExecutor(max_workers=24) as pool:
            for start in range(0,len(pending),24):
                group=pending[start:start+24];records=[json.loads((old/'episodes'/(name+'.json')).read_text()) for name in group]
                requests=[]
                for sim,name,r in zip(sims,group,records):
                    with np.load(old/'episodes'/(name+'.npz'),allow_pickle=False) as data:
                        actions=data['actions'].tolist();labels=data['stop_success'].copy()
                    n=len(actions)+(not actions or actions[-1]!=0)
                    eligible_positions=np.flatnonzero(labels[:n])
                    if len(eligible_positions):
                        anchors=sorted({int(eligible_positions[0]),int(eligible_positions[-1]),max(0,int(eligible_positions[0])-1)})
                    else:anchors=[max(0,n-1)]
                    seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
                    requests.append(pool.submit(sim.call,'replay',episode=by_id[name],actions=actions,anchors=anchors,seed=seed))
                responses=[f.result() for f in requests];images=[];bounds=[0]
                for r in responses:
                    images.extend(Image.fromarray(x) for x in r['images']);bounds.append(len(images))
                with gate.boundary(start):features=encoder.encode(images).cpu().numpy()
                for i,(name,r,response) in enumerate(zip(group,records,responses)):
                    ft=features[bounds[i]:bounds[i+1]];labels=response['stop_success']
                    assert len(ft)==len(labels)==len(response['root_step'])
                    path=folder/(name+'.npz');tmp=path.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp,features=ft,stop_success=labels,root_step=response['root_step'],
                        first_action=response['first_action'],second_action=response['second_action']);tmp.replace(path)
                    meta=dict(episode=name,scene=r['scene'],category=r['category'],domain=r['domain'],distance_group=r['distance_group'],
                        positions=len(ft),positive_stop_outcomes=int(labels.sum()),sha256=sha(path),
                        replay_source_sha256=sha(old/'episodes'/(name+'.npz')),training_only=True,
                        label='Existing environment binary STOP success only; no fixed symbolic ID',
                        roots=response['roots'],one_step_probes=response['one_step_probes'],two_step_probes=response['two_step_probes'],
                        no_coordinates_no_distances_no_collision_labels=True,no_hand_assigned_symbol_ids=True,
                        no_navigation_oracle=True,no_map=True,batch_image_encoding=256,runtime_feedback_input=False)
                    atomic(path.with_suffix('.json'),meta);done+=1;positions+=len(ft);positive+=int(labels.sum())
                state.update(state='collecting',completed=done,positions=positions,positive_stop_outcomes=positive,
                    seconds=time.time()-began,time=time.time(),gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                atomic(OUT/f'status_rank{args.rank}.json',state);print(json.dumps(state),flush=True)
        atomic(OUT/f'status_rank{args.rank}.json',dict(state,state='complete',time=time.time()))
    except Exception as e:
        atomic(OUT/f'status_rank{args.rank}.json',dict(state,state='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        gate.close()
        for sim in sims:sim.close()

if __name__=='__main__':main()
