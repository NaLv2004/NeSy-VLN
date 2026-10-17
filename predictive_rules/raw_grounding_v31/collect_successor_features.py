"""Recover unquantized visual evidence for existing training transitions.

Uses immutable recorded actions and privately probes their 17 possible
successor observations. This is training data preparation, not an evaluation.
No geometry, reward, collision label or privileged semantic state is exported.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from predictive_nesy.environment import FrozenVision,make_bank
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.data import sha
from evaluate_state512_v25 import atomic

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')


class Probe:
    def __init__(self,bank,out):
        sys.path.insert(0,'/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')
        from objectnav_rl.ipc import send,receive
        self.send,self.receive=send,receive
        self.log=(out/'simulator.log').open('ab')
        self.process=subprocess.Popen([bank.env['sim_python'],str(Path(__file__).with_name('successor_worker.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        self.call('init',experiment=bank.env,train_scenes=bank.provenance['train_layouts'])
    def call(self,op,**kwargs):
        self.send(self.process.stdin,dict(op=op,**kwargs))
        if not select.select([self.process.stdout],[],[],180)[0]:raise TimeoutError('Existing data-generation worker timed out')
        response=self.receive(self.process.stdout)
        if not response['ok']:raise RuntimeError(response['error'])
        return response['result']
    def close(self):
        if self.process.poll() is None:
            self.call('close');self.process.wait(timeout=15)
        self.log.close()


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES'] in {'3','4','5','6'}
    assert 0<=args.shard<args.shards
    out=Path(args.out).resolve();assert out.is_relative_to(ROOT);out.mkdir(exist_ok=True)
    logdir=out/f'shard_{args.shard}';logdir.mkdir(exist_ok=True)
    lock=(logdir/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    source=saved['manifest'];bank=make_bank()
    assert source['data']['train_layouts']==bank.provenance['train_layouts']
    jobs=[]
    # Recent proof-policy data first; all banks remain in the collection.
    for item in reversed(source['transition_banks']):
        folder=Path(item['path']);manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['training_only']
        for name in sorted(manifest['episodes']):
            path=folder/(name+'.npz')
            if path.exists():
                assert bank.by_id[name]['scene'] in bank.provenance['train_layouts']
                jobs.append(path)
    if args.shard==0:atomic(out/'manifest.json',dict(training_only=True,source_checkpoint=args.checkpoint,
        train_layouts=bank.provenance['train_layouts'],development_layouts=bank.provenance.get('dev_layouts'),
        expected_episodes=len(jobs),feature_shape=[49,128],feature_dtype='float16',
        data_fields=['features','successor_features','recorded_codes','recorded_successors','recorded_actions','probe_valid'],
        scene_geometry_used_only_by_simulator=True,semantic_state_labels=False))
    jobs=jobs[args.shard::args.shards]
    vision=FrozenVision(source['frozen_vision']);gate=NeuralBoundaryGate([vision.vision],interval=1)
    encoder=RawBoundaryGraphEncoder(vision,gate);sim=Probe(bank,logdir);began=time.time();completed=positions=0
    try:
        for path in jobs:
            target=out/path.parent.name/(path.stem+'.npz');target.parent.mkdir(exist_ok=True)
            if target.exists() and target.with_suffix('.json').exists():
                record=json.loads(target.with_suffix('.json').read_text())
                assert record['source_sha256']==sha(path) and record['output_sha256']==sha(target)
                completed+=1;positions+=record['positions'];continue
            with np.load(path,allow_pickle=False) as d:
                codes=d['codes'].copy();actions=d['actions'].copy();successors=d['successors'].copy();valid=d['probe_valid'].copy()
            sim.call('reset',episode=bank.by_id[path.stem]);raw=[];future=[]
            for t,action in enumerate(actions):
                response=sim.call('probe_and_execute',action=int(action))
                images=[Image.fromarray(response['current'])]+[Image.fromarray(x) for x in response['successors']]
                with gate.boundary(t):tokens,features=encoder.encode_joint(images)
                tokens=tokens.cpu().numpy().astype(np.uint8);features=features.cpu().numpy()
                if not np.array_equal(tokens[0],codes[t]):raise RuntimeError('Recorded current image mismatch: '+str(path))
                if valid[t] and not np.array_equal(tokens[1:],successors[t]):raise RuntimeError('Recorded successor image mismatch: '+str(path))
                if valid[t] and not response['probe_matches_execution']:raise RuntimeError('Probe/execute mismatch')
                raw.append(features[0]);future.append(features[1:])
            temp=target.with_suffix('.tmp.npz')
            np.savez_compressed(temp,features=np.stack(raw),successor_features=np.stack(future),
                recorded_codes=codes,recorded_successors=successors,recorded_actions=actions,probe_valid=valid)
            os.replace(temp,target)
            record=dict(bank=path.parent.name,episode=path.stem,source_sha256=sha(path),output_sha256=sha(target),positions=int(valid.sum()))
            atomic(target.with_suffix('.json'),record);completed+=1;positions+=int(valid.sum())
            status=dict(state='collecting',pid=os.getpid(),time=time.time(),completed=completed,total=len(jobs),
                positions=positions,seconds=time.time()-began,training_only=True)
            atomic(logdir/'status.json',status)
            if completed%5==0:print(json.dumps(status),flush=True)
        atomic(logdir/'status.json',dict(status,state='complete',time=time.time()))
    finally:gate.close();sim.close()


if __name__=='__main__':main()
