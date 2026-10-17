"""Overlap four independent training replays with one visual encoder.

The recorded trajectories, image checks, dataset split and saved features are
unchanged. Rendering is parallel; neural encoding remains on one thread.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from collect_successor_features import Probe,ROOT
from predictive_nesy.environment import FrozenVision,make_bank
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.data import sha
from evaluate_state512_v25 import atomic


def render(sim,episode,path):
    with np.load(path,allow_pickle=False) as source:
        data={key:source[key].copy() for key in ['codes','actions','successors','probe_valid']}
    sim.call('reset',episode=episode)
    frames=[sim.call('probe_and_execute',action=int(action)) for action in data['actions']]
    return data,frames


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6' and 1<=args.workers<=4
    out=Path(args.out).resolve();assert out.is_relative_to(ROOT)
    logdir=out/'parallel_render';logdir.mkdir(exist_ok=True)
    lock=(logdir/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    manifest=saved['manifest'];bank=make_bank()
    assert manifest['data']['train_layouts']==bank.provenance['train_layouts']
    jobs=[];completed=positions=0
    for record in reversed(manifest['transition_banks']):
        folder=Path(record['path']);source=json.loads((folder/'manifest.json').read_text())
        assert source['training_only']
        for name in sorted(source['episodes']):
            path=folder/(name+'.npz')
            if not path.exists():continue
            assert bank.by_id[name]['scene'] in bank.provenance['train_layouts']
            target=out/folder.name/path.name;target.parent.mkdir(exist_ok=True)
            if target.exists() and target.with_suffix('.json').exists():
                previous=json.loads(target.with_suffix('.json').read_text())
                assert previous['source_sha256']==sha(path) and previous['output_sha256']==sha(target)
                completed+=1;positions+=previous['positions']
            else:jobs.append((path,target))
    total=completed+len(jobs);began=time.time()
    status=dict(state='starting',pid=os.getpid(),time=began,completed=completed,total=total,positions=positions,
        rendering_workers=args.workers,training_only=True)
    atomic(logdir/'status.json',status)
    vision=FrozenVision(manifest['frozen_vision']);gate=NeuralBoundaryGate([vision.vision],interval=1)
    encoder=RawBoundaryGraphEncoder(vision,gate);sims=[]
    try:
        for i in range(args.workers):
            folder=logdir/f'worker_{i}';folder.mkdir(exist_ok=True);sims.append(Probe(bank,folder))
        pending={};iterator=iter(jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            def dispatch(worker):
                item=next(iterator,None)
                if item is not None:
                    path,target=item
                    future=pool.submit(render,sims[worker],bank.by_id[path.stem],path)
                    pending[future]=(worker,path,target)
            for i in range(len(sims)):dispatch(i)
            while pending:
                ready,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in ready:
                    worker,path,target=pending.pop(future);data,frames=future.result()
                    dispatch(worker)
                    raw=[];successors=[]
                    for t,response in enumerate(frames):
                        images=[Image.fromarray(response['current'])]+[Image.fromarray(rgb) for rgb in response['successors']]
                        with gate.boundary(t):tokens,features=encoder.encode_joint(images)
                        tokens=tokens.cpu().numpy().astype(np.uint8);features=features.cpu().numpy()
                        if not np.array_equal(tokens[0],data['codes'][t]):raise RuntimeError('Recorded image mismatch: '+str(path))
                        if data['probe_valid'][t]:
                            if not np.array_equal(tokens[1:],data['successors'][t]):raise RuntimeError('Recorded successor mismatch: '+str(path))
                            if not response['probe_matches_execution']:raise RuntimeError('Probe/execute mismatch')
                        raw.append(features[0]);successors.append(features[1:])
                    temporary=target.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary,features=np.stack(raw),successor_features=np.stack(successors),
                        recorded_codes=data['codes'],recorded_successors=data['successors'],recorded_actions=data['actions'],probe_valid=data['probe_valid'])
                    os.replace(temporary,target)
                    count=int(data['probe_valid'].sum())
                    atomic(target.with_suffix('.json'),dict(bank=path.parent.name,episode=path.stem,source_sha256=sha(path),output_sha256=sha(target),positions=count))
                    completed+=1;positions+=count
                    status.update(state='collecting',time=time.time(),completed=completed,positions=positions,seconds=time.time()-began)
                    atomic(logdir/'status.json',status)
                    if completed%5==0:print(json.dumps(status),flush=True)
        atomic(logdir/'status.json',dict(status,state='complete',time=time.time()))
    except Exception as error:
        import traceback
        atomic(logdir/'status.json',dict(status,state='failed',time=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        gate.close()
        for sim in sims:sim.close()


if __name__=='__main__':main()
