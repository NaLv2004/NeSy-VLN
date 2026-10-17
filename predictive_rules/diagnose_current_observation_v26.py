"""Diagnostic only: isolate fresh hard observation IDs with frozen v25 rules.

This is not the final adaptive-refresh model. It keeps the old ten-step
schedule solely to measure the effect of the newly permitted observation
feedback without conflating it with retraining or a new renewal policy.
"""
import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from predictive_nesy.data import sha
from predictive_nesy.environment import FrozenVision, make_bank
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.state512_rule_world import State512RuleWorld, Config, export_program
from predictive_nesy.graph_subgoal_interpreter import plan
from evaluate_state512_v25 import atomic, summarize


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    assert not out.exists(),'Preserve completed diagnostic outputs'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    model=State512RuleWorld(Config(**saved['model_config'])).cuda().eval()
    model.load_state_dict(saved['model']);model.requires_grad_(False)
    bank=make_bank();episodes=bank.dev
    assert len(episodes)==80 and not set(e['scene'] for e in episodes)&set(bank.provenance['train_layouts'])
    atomic(out/'manifest.json',dict(diagnostic_only=True,final_adaptive_refresh_requirement_not_yet_tested=True,
        checkpoint_sha256=sha(args.checkpoint),episodes=[e['episode_id'] for e in episodes],
        data=bank.provenance,observation_interface='one integer in [0,511] per step; external label = ID+1',
        feedback='replace stale predicted start state with the current encoded state; exact goal unchanged',
        rule_revision_interval=10,training_or_optimizer_steps=0))
    vision=FrozenVision(saved['manifest']['frozen_vision'])
    gate=NeuralBoundaryGate([model,vision.vision],interval=1)
    encoder=RawBoundaryGraphEncoder(vision,gate)
    rows=[];sims=[];began=time.time()
    try:
        sims=[bank.simulator(out/f'sim_{i}.log',0,evaluation=True) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0,80,4):
                batch=episodes[offset:offset+4]
                responses=list(pool.map(lambda p:p[0].call('reset',episode=p[1]),zip(sims,batch)))
                images=[[Image.fromarray(np.asarray(r['rgb'])[...,:3].astype(np.uint8))] for r in responses]
                histories=[[] for _ in batch];raws=[[] for _ in batch];actions=[[17] for _ in batch]
                programs={};perception={};previous_prediction={}
                records=[dict(episode=e,programs=[],steps=[],metrics={},planning_failed=False,
                              boundary_seconds=0.,rule_step_seconds=0.) for e in batch]
                tasks=torch.tensor([['chair','table'].index(e['category']) for e in batch],device='cuda')
                active=list(range(len(batch)))
                for step in range(32):
                    begin=time.perf_counter()
                    with gate.boundary(step):
                        for i in active:
                            code,raw=encoder.encode_joint([images[i][-1]])
                            histories[i].append(code[0]);raws[i].append(raw[0])
                        if step%10==0:
                            width=min(step+1,11)
                            history=torch.stack([torch.stack(histories[i][-width:]) for i in active])
                            raw=torch.stack([torch.stack(raws[i][-width:]) for i in active])
                            prior=torch.tensor([actions[i][-width:] for i in active],device='cuda')
                            length=torch.full((len(active),),width,device='cuda',dtype=torch.long)
                            generated=model.generate(tasks[active],history,prior,length,start=step,boundary_features=raw)
                            observed=model.ground(generated,history[:,-1]).argmax(-1).flatten().cpu().tolist()
                            for j,i in enumerate(active):
                                programs[i]=export_program(generated,j)
                                perception[i]={k:generated[k][j:j+1].clone() for k in ['definitions','ground_memory','ground_last','ground_codes']}
                                records[i]['programs'].append(dict(start=step,digest=programs[i].digest(),goal=programs[i].subgoal.tolist()))
                            del generated
                        else:
                            observed=[]
                            for i in active:
                                p=perception[i];code=histories[i][-1][None]
                                feature=model.observation(model.embed(code))
                                memory,last,old_codes=model.advance(p['ground_memory'],p['ground_last'],p['ground_codes'],feature,code)
                                p.update(ground_memory=memory,ground_last=last,ground_codes=old_codes)
                                observed.append(int(model.ground(p,code).argmax(-1)))
                    duration=time.perf_counter()-begin
                    requests=[]
                    for j,i in enumerate(active):
                        records[i]['boundary_seconds']+=duration/len(active)
                        # The planner receives one integer only, plus its
                        # immutable rule arrays. No perception feature escapes.
                        current=int(observed[j]);remaining=min(10-step%10,32-step)
                        t=time.perf_counter();proof=plan(programs[i],np.array([current],np.int64),horizon=remaining)
                        records[i]['rule_step_seconds']+=time.perf_counter()-t
                        if not proof['found']:
                            records[i]['planning_failed']=True
                            records[i]['failed_plan']=dict(step=step,observed=current,goal=programs[i].subgoal.tolist())
                            continue
                        action=int(proof['actions'][0]);after=int(programs[i].transitions[current,action].argmax())
                        records[i]['steps'].append(dict(step=step,observed_id=current,
                            prior_prediction=previous_prediction.get(i),action=action,after=after,plan=proof,
                            program_sha256=programs[i].digest()))
                        previous_prediction[i]=after;requests.append((i,action))
                    if not requests:break
                    responses=list(pool.map(lambda p:sims[p[0]].call('step',action=p[1]),requests));retained=[]
                    for (i,action),response in zip(requests,responses):
                        images[i].append(Image.fromarray(np.asarray(response['rgb'])[...,:3].astype(np.uint8)))
                        actions[i].append(action);records[i]['metrics']=response.get('metrics',{})
                        records[i]['steps'][-1]['collision_for_report_only']=bool(response.get('feedback',{}).get('collision',False))
                        if not response['done']:retained.append(i)
                    active=retained
                    if not active:break
                for e,record in zip(batch,records):
                    folder=out/e['episode_id'];folder.mkdir()
                    atomic(folder/'trace.json',record)
                    row=dict(episode_id=e['episode_id'],scene=e['scene'],domain=e['domain'],distance_group=e['distance_group'],category=e['category'],
                        success=bool(record['metrics'].get('success',False)) and not record['planning_failed'],steps=len(record['steps']),
                        collisions=sum(r['collision_for_report_only'] for r in record['steps']),planning_failed=record['planning_failed'],
                        program_count=len(record['programs']),proven_plan_count=len(record['steps']),
                        boundary_seconds=record['boundary_seconds'],rule_step_seconds=record['rule_step_seconds'])
                    rows.append(row);atomic(folder/'evaluation.json',row)
                atomic(out/'status.json',dict(state='evaluating',pid=os.getpid(),completed=len(rows),successes=sum(r['success'] for r in rows),time=time.time()))
                print(json.dumps(dict(completed=len(rows),successes=sum(r['success'] for r in rows))),flush=True)
        # summarize's old plan_attempt field refers to revisions; record the
        # new per-action planning quantities explicitly and remove old ratio.
        result=summarize(rows)
        for key in ['exact_plan_found_rate','neural_calls_outside_boundaries']:result.pop(key,None)
        result.update(diagnostic_only=True,seconds=time.time()-began,per_step_neural_observation_allowed=True,
            every_planner_observation_is_one_hard_id=True,success_criterion='unchanged development80 simulator success',
            executed_steps=sum(r['steps'] for r in rows),rules_generated=sum(r['program_count'] for r in rows),
            executed_steps_per_program=sum(r['steps'] for r in rows)/max(sum(r['program_count'] for r in rows),1))
        atomic(out/'summary.json',result);atomic(out/'status.json',dict(state='complete',time=time.time(),summary=result))
        print(json.dumps(result),flush=True)
    finally:
        gate.close()
        for sim in sims:sim.close()


if __name__=='__main__':main()
