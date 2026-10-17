"""Actual simulator action outcomes on training layouts, for world-model labels.

The demonstrator follows a pre-existing action list. Probe images and metrics
cannot change that list. No oracle, map, pose or goal coordinate enters a policy.
"""
import argparse,fcntl,json,os,random,select,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from predictive_nesy.data import BASE,read,sha
from predictive_nesy.environment import load_previous,make_bank,FrozenVision,dependencies
from train import atomic




class CollectorSimulator:
    def __init__(self,bank,out,worker='counterfactual_worker.py'):
        sys.path.insert(0,'/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')
        from objectnav_rl.ipc import receive,send
        self.receive,self.send=receive,send
        self.log=(out/'simulator.log').open('ab')
        env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if worker not in {'counterfactual_worker.py','anonymous_counterfactual_worker.py'}:raise ValueError('Unknown training worker')
        self.process=subprocess.Popen([bank.env['sim_python'],str(Path(__file__).with_name(worker))],env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        self.call('init',experiment=bank.env,train_scenes=bank.provenance['train_layouts'])

    def call(self,operation,**kwargs):
        self.send(self.process.stdin,dict(op=operation,**kwargs))
        if not select.select([self.process.stdout],[],[],180)[0]:raise TimeoutError('Training collector response timeout; do not relaunch while its PID is live')
        value=self.receive(self.process.stdout)
        if not value['ok']:raise RuntimeError(value['error'])
        return value['result']

    def close(self):
        if self.process.poll() is None:
            try:self.call('close')
            finally:self.process.wait(timeout=15)
        self.log.close()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=96);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='7':raise RuntimeError('Only the assigned GPU 7 is authorized')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'collect.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    _,manifest,commit=load_previous()
    # The previous actor is not used. Only its already-pinned perceptual assets.
    del _;torch.cuda.empty_cache()
    bank=make_bank();vision=FrozenVision(manifest['source_commit'])
    rows=read(BASE/'perceptual_sequences_v73/tail_1e4/training_rows.json')
    groups={}
    for row in rows:
        if row['episode_id'] in bank.by_id and all(part not in row['episode_id'] for part in ['paired','recovery']):
            groups.setdefault(row['episode_id'],[]).append(int(row['label']))
    rng=random.Random(1517);strata={}
    for episode in bank.train:
        if episode['episode_id'] in groups:
            key=(episode['domain'],episode['distance_group'],episode['category'])
            strata.setdefault(key,[]).append(episode)
    for values in strata.values():rng.shuffle(values)
    selected=[];keys=sorted(strata)
    while len(selected)<a.episodes:
        changed=False
        for key in keys:
            if strata[key] and len(selected)<a.episodes:selected.append(strata[key].pop());changed=True
        if not changed:break
    record=dict(training_only=True,episodes=[e['episode_id'] for e in selected],data=bank.provenance,
        source_model_commit=commit,source_rows_sha256=sha(BASE/'perceptual_sequences_v73/tail_1e4/training_rows.json'),
        action_source='pre-existing demonstration; fixed before probes',observations='labels only',
        features=['visual_patch_codes','executed_action','successor_visual_patch_codes','collision','goal_eligible'],
        forbidden_features_not_saved=['map','pose','goal_coordinates','distance'])
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==record
    else:atomic(out/'manifest.json',record)
    simulator=CollectorSimulator(bank,out);begin=time.time();done=0;total_states=0;matched=0
    try:
        for episode in selected:
            target=out/(episode['episode_id']+'.npz')
            if target.exists():done+=1;continue
            simulator.call('reset',episode=episode)
            actions=groups[episode['episode_id']][:32]
            trace=[]
            for action in actions:
                trace.append(simulator.call('probe_and_execute',action=action))
                if trace[-1]['done']:break
            # Encoding occurs after the entire preset action sequence finishes.
            # It cannot revise an already executed demonstration action.
            cache={}
            import hashlib
            def encode(rgb):
                digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).digest()
                if digest not in cache:cache[digest]=vision.encode([Image.fromarray(rgb)])[0].cpu().numpy().astype(np.uint8)
                return cache[digest]
            current=np.stack([encode(r['current']) for r in trace])
            successors=np.stack([np.stack([encode(rgb) for rgb in r['successors']]) for r in trace])
            actual=np.stack([encode(r['actual']) for r in trace])
            temporary=target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,codes=current,successors=successors,actual=actual,
                actions=np.array([r['action'] for r in trace],np.int64),
                collision=np.array([r['collision'] for r in trace],bool),
                eligible_next=np.array([r['eligible_next'] for r in trace],bool),
                eligible_current=np.array([r['eligible_current'] for r in trace],bool),
                probe_valid=np.array([r['probe_matches_execution'] for r in trace],bool),
                probe_rgb_error=np.array([r['probe_rgb_error'] for r in trace],np.float32),
                task=np.array(['chair','table'].index(episode['category'])))
            os.replace(temporary,target)
            done+=1;total_states+=len(trace);matched+=sum(r['probe_matches_execution'] for r in trace)
            state=dict(state='collecting',pid=os.getpid(),simulator_pid=simulator.process.pid,completed=done,total=len(selected),
                states_this_run=total_states,matched_probes_this_run=matched,time=time.time(),seconds=time.time()-begin)
            atomic(out/'status.json',state);print(json.dumps(state),flush=True)
        atomic(out/'COMPLETE.json',dict(episodes=done,time=time.time(),manifest_sha256=sha(out/'manifest.json')))
        atomic(out/'status.json',dict(state='complete',pid=os.getpid(),completed=done,total=len(selected),time=time.time()))
    finally:simulator.close()


if __name__=='__main__':
    main()
