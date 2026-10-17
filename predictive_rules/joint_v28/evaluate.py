"""Unchanged development80, hard feedback and learned asynchronous renewal."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from joint_model import JointObservedWorld as AdaptiveObservedWorld, Config
from hard_interpreter import export, plan, Program, corrected_state
from predictive_nesy.data import sha
from predictive_nesy.environment import FrozenVision, make_bank
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from evaluate_state512_v25 import atomic


def save_program(path,program):
    np.savez_compressed(path,next_state=program.next_state,confidence=program.confidence,
                        correction=program.correction,goal=program.goal,start=program.start)


def audit(record,folder):
    loaded={}
    for event in record['programs']:
        with np.load(folder/event['file'],allow_pickle=False) as data:
            p=Program(data['next_state'],data['confidence'],data['correction'],int(data['goal']),int(data['start'])).validate()
        assert p.digest()==event['digest'];loaded[event['digest']]=p
    for step in record['steps']:
        p=loaded[step['program_sha256']];proof=step['proof']
        assert type(step['observed_id']) is int and 0<=step['observed_id']<512
        current=corrected_state(p,proof['predicted'],step['observed_id'])
        assert current==proof['corrected']==proof['states'][0]
        assert proof['actions'][0]==step['action']
        for j,action in enumerate(proof['actions']):
            current=int(p.next_state[current,action]);assert current==proof['states'][j+1]
            if action==0:assert j==len(proof['actions'])-1
        assert current==p.goal
    return True


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    assert saved['manifest']['architecture']=='joint_hard_observation_v28'
    model=AdaptiveObservedWorld(Config(**saved['model_config'])).cuda().eval()
    model.load_state_dict(saved['model']);model.requires_grad_(False)
    bank=make_bank();episodes=bank.dev
    assert len(episodes)==80 and not set(e['scene'] for e in episodes)&set(bank.provenance['train_layouts'])
    atomic(out/'manifest.json',dict(architecture=saved['manifest']['architecture'],checkpoint_sha256=sha(args.checkpoint),
        episode_ids=[e['episode_id'] for e in episodes],data=bank.provenance,max_actions=32,
        every_step_perception_boundary='one hard observation ID, 0-based internally / 1..512 externally',
        adaptive_revision='learned GRU probability >=0.5; first program required; no fixed-period renewal',
        no_goal_replacement=True,no_direct_action_head=True,no_neural_actor_fallback=True,
        no_map_or_simulator_feedback_to_policy=True))
    frontend=FrozenVision(saved['manifest']['frozen_vision'])
    gate=NeuralBoundaryGate([model,frontend.vision],interval=1)
    encoder=RawBoundaryGraphEncoder(frontend,gate)
    counts={'generation_forward':0,'inverse_calls':0}
    def count_generation(module,inputs):counts['generation_forward']+=1
    handle=model.subgoal.register_forward_pre_hook(count_generation)
    def reject_inverse(module,inputs):
        counts['inverse_calls']+=1
        raise RuntimeError('The training-only inverse action decoder cannot run during navigation')
    inverse_handle=model.inverse_discrete.register_forward_pre_hook(reject_inverse)
    sims=[];rows=[];all_gaps=[];began=time.time()
    try:
        sims=[bank.simulator(out/f'sim_{i}.log',0,evaluation=True) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0,80,4):
                batch=episodes[offset:offset+4]
                responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(sims,batch)))
                images=[[Image.fromarray(np.asarray(r['rgb'])[...,:3].astype(np.uint8))] for r in responses]
                histories=[[] for _ in batch];raws=[[] for _ in batch];actions=[[17] for _ in batch]
                programs={};perceptions={};predictions={};hidden={i:torch.zeros(1,128,device='cuda') for i in range(len(batch))}
                records=[dict(episode=e,programs=[],steps=[],renewal_decisions=[],metrics={},planning_failed=False) for e in batch]
                tasks=[torch.tensor([['chair','table'].index(e['category'])],device='cuda') for e in batch]
                active=list(range(len(batch)))
                for step in range(32):
                    requests=[]
                    for i in active:
                        with gate.boundary(step):
                            code,raw=encoder.encode_joint([images[i][-1]])
                            histories[i].append(code[0]);raws[i].append(raw[0])
                            observed=None;renew=step==0;probability=1.
                            if step:
                                perception=perceptions[i]
                                feature=model.observation(model.embed(code))
                                memory,last,old_codes=model.advance(perception['ground_memory'],perception['ground_last'],perception['ground_codes'],feature,code)
                                perception.update(ground_memory=memory,ground_last=last,ground_codes=old_codes)
                                observed=int(model.ground(perception,code).argmax(-1))
                                value,hidden[i]=model.revision.step(torch.tensor([observed],device='cuda'),
                                    torch.tensor([predictions[i]],device='cuda'),torch.tensor([actions[i][-1]],device='cuda'),
                                    tasks[i],torch.tensor([step-programs[i].start],device='cuda'),hidden[i],
                                    goal=torch.tensor([programs[i].goal],device='cuda'))
                                probability=float(value.sigmoid());renew=probability>=.5
                            if renew:
                                width=min(step+1,11)
                                history=torch.stack(histories[i][-width:])[None];raw_history=torch.stack(raws[i][-width:])[None]
                                previous=torch.tensor([actions[i][-width:]],device='cuda');length=torch.tensor([width],device='cuda')
                                generated=model.generate(tasks[i],history,previous,length,start=step,boundary_features=raw_history)
                                program=export(generated);programs[i]=program
                                perceptions[i]={key:generated[key].clone() for key in ['definitions','ground_memory','ground_last','ground_codes']}
                                observed=int(model.ground(generated,code).argmax(-1));predictions[i]=observed
                                folder=out/batch[i]['episode_id'];folder.mkdir(parents=True,exist_ok=True)
                                name=f'program_{step:02d}.npz';save_program(folder/name,program)
                                if records[i]['programs']:all_gaps.append(step-records[i]['programs'][-1]['start'])
                                records[i]['programs'].append(dict(start=step,file=name,digest=program.digest(),goal=program.goal))
                                # Gate training replays from the retained program's
                                # boundary. Reset at every renewal, not just step0.
                                hidden[i]=torch.zeros_like(hidden[i])
                                _,hidden[i]=model.revision.step(torch.tensor([observed],device='cuda'),torch.tensor([observed],device='cuda'),
                                    torch.tensor([actions[i][-1]],device='cuda'),tasks[i],torch.tensor([0],device='cuda'),hidden[i],
                                    goal=torch.tensor([program.goal],device='cuda'))
                                del generated
                        records[i]['renewal_decisions'].append(dict(step=step,probability=probability,generated=renew))
                        proof=plan(programs[i],int(predictions[i]),int(observed),horizon=min(10,32-step))
                        if not proof['found']:
                            records[i]['planning_failed']=True;records[i]['failed_proof']=proof;continue
                        action=int(proof['actions'][0]);predictions[i]=int(proof['states'][1])
                        records[i]['steps'].append(dict(step=step,observed_id=int(observed),observed_external_id=int(observed)+1,
                            action=action,proof=proof,program_sha256=programs[i].digest()))
                        requests.append((i,action))
                    if not requests:break
                    responses=list(pool.map(lambda pair:sims[pair[0]].call('step',action=pair[1]),requests));retained=[]
                    for (i,action),response in zip(requests,responses):
                        actions[i].append(action);images[i].append(Image.fromarray(np.asarray(response['rgb'])[...,:3].astype(np.uint8)))
                        records[i]['metrics']=response.get('metrics',{})
                        records[i]['steps'][-1]['collision_for_report_only']=bool(response.get('feedback',{}).get('collision',False))
                        if not response['done']:retained.append(i)
                    active=retained
                    if not active:break
                for i,(episode,record) in enumerate(zip(batch,records)):
                    folder=out/episode['episode_id'];folder.mkdir(parents=True,exist_ok=True)
                    audit(record,folder);atomic(folder/'trace.json',record)
                    images[i][0].save(folder/'first.png');images[i][-1].save(folder/'last.png')
                    rows.append(dict(episode_id=episode['episode_id'],domain=episode['domain'],distance_group=episode['distance_group'],category=episode['category'],
                        success=bool(record['metrics'].get('success',False)) and not record['planning_failed'],
                        steps=len(record['steps']),programs=len(record['programs']),planning_failed=record['planning_failed'],
                        collisions=sum(s['collision_for_report_only'] for s in record['steps'])))
                atomic(out/'status.json',dict(state='evaluating',pid=os.getpid(),completed=len(rows),successes=sum(r['success'] for r in rows),time=time.time()))
                print(json.dumps(dict(completed=len(rows),successes=sum(r['success'] for r in rows))),flush=True)
        total_steps=sum(r['steps'] for r in rows);programs=sum(r['programs'] for r in rows)
        assert counts['generation_forward']==programs
        groups=defaultdict(list)
        for row in rows:
            for field in ['domain','distance_group','category']:groups[field+':'+row[field]].append(row)
        result=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows),
            executed_steps=total_steps,rules_generated=programs,executed_steps_per_program=total_steps/max(programs,1),
            mean_between_regeneration_steps=float(np.mean(all_gaps)) if all_gaps else None,
            five_step_average_requirement_met=total_steps>=5*programs,
            development_sixty_percent_requirement_met=sum(r['success'] for r in rows)>=48,
            planning_failures=sum(r['planning_failed'] for r in rows),collision_action_rate=sum(r['collisions'] for r in rows)/max(total_steps,1),
            all_actions_independently_audited_hard_goal_proofs=True,neural_generation_calls_verified=counts['generation_forward'],
            training_only_inverse_calls=counts['inverse_calls'],
            by_group={key:dict(episodes=len(value),successes=sum(r['success'] for r in value),success_rate=sum(r['success'] for r in value)/len(value)) for key,value in groups.items()},
            seconds=time.time()-began,checkpoint_sha256=sha(args.checkpoint))
        atomic(out/'episodes.json',rows);atomic(out/'summary.json',result);atomic(out/'status.json',dict(state='complete',time=time.time(),summary=result))
        print(json.dumps(result),flush=True)
    finally:
        inverse_handle.remove();handle.remove();gate.close()
        for sim in sims:sim.close()


if __name__=='__main__':main()
