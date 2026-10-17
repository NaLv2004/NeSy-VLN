"""Collect private labels along a model's own ten-action symbolic plans.

The policy only reads frames at revision boundaries. All counterfactual probes
are training targets and cannot change a submitted action or intermediate state.
Two shards can collect different training episodes on separately assigned GPUs.
"""
import argparse
import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from collect_counterfactuals import CollectorSimulator
from predictive_nesy.data import BASE, read, sha
from predictive_nesy.environment import FrozenVision, make_bank
from predictive_nesy.model_memory import PredictiveMemoryModel, MemoryConfig
from predictive_nesy.interpreter import export_program
from predictive_nesy.belief_interpreter import BeliefController
from predictive_nesy.gate import NeuralBoundaryGate
from train import atomic


def select_training(bank, count):
    rng=random.Random(1527)
    groups={}
    for episode in bank.train:
        if episode['scene'] not in bank.provenance['train_layouts']:
            raise ValueError('Nontraining layout')
        key=(episode['domain'],episode['distance_group'],episode['category'])
        groups.setdefault(key,[]).append(episode)
    for values in groups.values():rng.shuffle(values)
    selected=[]
    while len(selected)<count:
        added=0
        for key in sorted(groups):
            if groups[key] and len(selected)<count:
                selected.append(groups[key].pop());added+=1
        if not added:break
    return selected


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--shards',type=int,default=2);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in {'3','4','5','7'}:raise RuntimeError('GPU not authorized')
    if not 0<=a.shard<a.shards:raise ValueError('Invalid shard')
    out=Path(a.out).resolve()
    if not out.is_relative_to(BASE/'predictive_rules_20261015'):raise ValueError('Output escaped root')
    out.mkdir(parents=True,exist_ok=True)
    logs=out/f'shard_{a.shard}';logs.mkdir(exist_ok=True)
    lock=(logs/'collect.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    bank=make_bank();selected=select_training(bank,a.episodes)
    if bank.provenance['train_layouts']!=saved['manifest']['data']['train_layouts']:
        raise ValueError('Checkpoint training split changed')
    record=dict(training_only=True,episodes=[e['episode_id'] for e in selected],data=bank.provenance,
        policy_checkpoint=dict(path=a.checkpoint,sha256=sha(a.checkpoint),update=saved['update']),
        action_source='ten-action symbolic beam plans from a frozen policy; no intermediate observation correction',
        revision_interval=10,horizon=32,observations='counterfactual labels never enter planning',
        features=['visual_patch_codes','executed_action','successor_visual_patch_codes','collision','goal_eligible'],
        forbidden_features_not_saved=['map','pose','goal_coordinates','distance'])
    with (out/'manifest.lock').open('a') as manifest_lock:
        fcntl.flock(manifest_lock,fcntl.LOCK_EX)
        if (out/'manifest.json').exists():assert read(out/'manifest.json')==record
        else:atomic(out/'manifest.json',record)
    if a.prepare_only:
        print(json.dumps(dict(prepared=True,episodes=len(selected),manifest_sha256=sha(out/'manifest.json'))));return
    m=PredictiveMemoryModel(MemoryConfig(**saved['model_config'])).cuda().eval()
    m.load_state_dict(saved['model']);source=saved['manifest']['frozen_vision'];del saved
    vision=FrozenVision(source);sim=CollectorSimulator(bank,logs)
    assigned=selected[a.shard::a.shards];begin=time.time();done=0;states=0
    try:
        for episode in assigned:
            target=out/(episode['episode_id']+'.npz')
            if target.exists():done+=1;continue
            initial=sim.call('reset',episode=episode)
            images=[Image.fromarray(initial['rgb'])];actions=[17]
            controller=BeliefController();trace=[];program_records=[];proofs=[]
            folder=out/'proofs'/episode['episode_id'];folder.mkdir(parents=True,exist_ok=True)
            gate=NeuralBoundaryGate([m,vision.vision])
            try:
                for step in range(32):
                    if step%10==0:
                        with gate.boundary(step):
                            codes=vision.encode(images[-11:])[None]
                            previous=torch.tensor([actions[-11:]],device='cuda')
                            task=torch.tensor([['chair','table'].index(episode['category'])],device='cuda')
                            generated=m.generate(task,codes,previous,torch.tensor([codes.shape[1]],device='cuda'),start=step)
                            state=m.ground(generated,codes[:,-1]).argmax(-1)[0].cpu().numpy()
                        program=export_program(generated,0)
                        plan=controller.revise(program,state,horizon=min(10,32-step))
                        filename=f'program_{step:02d}.npz';program.save(folder/filename)
                        program_records.append(dict(start=step,initial_state=state.tolist(),plan=plan,file=filename,
                            observed_frame_indices=list(range(max(0,step-10),step+1))))
                        del generated
                    action,proof=controller.next_action()
                    # The complete action is fixed before private probes start.
                    result=sim.call('probe_and_execute',action=action)
                    trace.append(result);proofs.append(proof)
                    images.append(Image.fromarray(result['actual']));actions.append(action)
                    if result['done']:break
                if gate.violations:raise RuntimeError('Unexpected within-block neural computation')
                atomic(folder/'trace.json',dict(episode=episode,steps=proofs,programs=program_records,
                    neural_calls_outside_boundaries=gate.violations,training_labels_only=True))
            finally:gate.close()
            # The rollout has ended. Future/probe images are encoded as labels
            # now, separately from the finished policy execution.
            cache={}
            def encode(rgb):
                digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).digest()
                if digest not in cache:
                    cache[digest]=vision.encode([Image.fromarray(rgb)])[0].cpu().numpy().astype(np.uint8)
                return cache[digest]
            temporary=target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,codes=np.stack([encode(r['current']) for r in trace]),
                successors=np.stack([np.stack([encode(rgb) for rgb in r['successors']]) for r in trace]),
                actual=np.stack([encode(r['actual']) for r in trace]),
                actions=np.array([r['action'] for r in trace],np.int64),
                collision=np.array([r['collision'] for r in trace],bool),
                eligible_next=np.array([r['eligible_next'] for r in trace],bool),
                eligible_current=np.array([r['eligible_current'] for r in trace],bool),
                probe_valid=np.array([r['probe_matches_execution'] for r in trace],bool),
                probe_rgb_error=np.array([r['probe_rgb_error'] for r in trace],np.float32),
                task=np.array(['chair','table'].index(episode['category'])))
            os.replace(temporary,target);done+=1;states+=len(trace)
            status=dict(state='collecting',pid=os.getpid(),gpu=int(gpu),completed=done,total=len(assigned),
                states_this_run=states,time=time.time(),seconds=time.time()-begin,neural_boundary_violations=0)
            atomic(logs/'status.json',status);print(json.dumps(status),flush=True)
        atomic(logs/'COMPLETE.json',dict(completed=done,total=len(assigned),time=time.time()))
        atomic(logs/'status.json',dict(state='complete',pid=os.getpid(),gpu=int(gpu),completed=done,time=time.time()))
    finally:sim.close()


if __name__=='__main__':main()
