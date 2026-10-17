"""Evaluate exact symbolic subgoal plans on the unchanged isolated tasks."""
import argparse,json,os,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from predictive_nesy.data import read,sha
from predictive_nesy.environment import FrozenVision,make_bank
from predictive_nesy.gate import NeuralBoundaryGate
from fast_vision import BoundaryGraphEncoder
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel as SubgoalRuleModel,Config
from predictive_nesy.graph_subgoal_interpreter import Controller,Program,export_program,successors


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temporary,path)


def audit(record,folder):
    proven={};count=0
    for event in record['programs']:
        p=Program.load(folder/event['file']);plan=event['plan'];state=np.asarray(event['initial_state']);start=event['start']
        assert event['observed_frame_indices']==list(range(max(0,start-10),start+1))
        assert p.start==start and p.digest()==plan['program_sha256']
        if not plan['found']:
            assert not plan['actions'];continue
        assert plan['actions'] and len(plan['states'])==len(plan['actions'])+1
        assert 0 not in plan['actions'][:-1]
        assert len(plan['actions'])==min(10,32-start) or plan['actions'][-1]==0
        for i,action in enumerate(plan['actions']):
            assert state.tolist()==plan['states'][i]
            state=successors(p,state)[0][action]
            assert state.tolist()==plan['states'][i+1]
            proven[start+i]=(action,plan['states'][i],state.tolist(),p.digest())
        assert np.array_equal(state,p.subgoal);count+=1
    for r in record['steps']:
        assert proven[r['step']]==(r['action'],r['before'],r['after'],r['program_sha256'])
    return count


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        for key in ['domain','distance_group','category']:groups[key+':'+row[key]].append(row)
        if row['domain']=='normal' and row['distance_group'] in ['near','medium']:groups['normal_near_medium'].append(row)
    stats={key:dict(episodes=len(v),successes=sum(x['success'] for x in v),success_rate=sum(x['success'] for x in v)/len(v),
        collision_action_rate=sum(x['collisions'] for x in v)/max(sum(x['steps'] for x in v),1)) for key,v in groups.items()}
    attempts=sum(r['program_count'] for r in rows);found=sum(r['proven_plan_count'] for r in rows)
    return dict(**stats['all'],by_group=stats,mean_steps=float(np.mean([r['steps'] for r in rows])),
        planning_failures=sum(r['planning_failed'] for r in rows),plan_attempts=attempts,proven_plans=found,
        exact_plan_found_rate=found/max(attempts,1),rule_step_seconds=sum(r['rule_step_seconds'] for r in rows),
        boundary_seconds=sum(r['boundary_seconds'] for r in rows),neural_calls_outside_boundaries=0)


@torch.no_grad()
def evaluate(model,manifest,bank,out,suite='development',beam_width=128):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);episodes=bank.dev if suite=='development' else bank.fixed
    assert len(episodes)==80 and not set(e['scene'] for e in episodes)&set(bank.provenance['train_layouts'])
    atomic(out/'manifest.json',dict(suite=suite,episode_ids=[e['episode_id'] for e in episodes],data=bank.provenance,
        horizon=32,revision_interval=10,planner='complete_finite_state_subgoal_search',state_count=model.cfg.states,
        no_reward_table=True,no_approximate_fallback=True,vision_encoding='single_frame_cuda_graph'))
    vision=FrozenVision(manifest['frozen_vision']);model.eval();gate=NeuralBoundaryGate([model,vision.vision]);encoder=BoundaryGraphEncoder(vision,gate);sims=[];rows=[];began=time.time()
    try:
        sims=[bank.simulator(out/f'sim_{i}.log',0,evaluation=True) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0,len(episodes),4):
                batch=episodes[offset:offset+4]
                responses=list(pool.map(lambda p:p[0].call('reset',episode=p[1]),zip(sims,batch)))
                images=[[Image.fromarray(np.asarray(r['rgb'])[...,:3].astype(np.uint8))] for r in responses]
                actions=[[17] for _ in batch];controllers=[Controller() for _ in batch]
                records=[dict(episode=e,steps=[],programs=[],metrics={},planning_failed=False,boundary_seconds=0.,rule_step_seconds=0.) for e in batch]
                tasks=torch.tensor([['chair','table'].index(e['category']) for e in batch],device='cuda');active=list(range(len(batch)))
                for step in range(32):
                    if step%10==0:
                        begin=time.perf_counter();width=min(step+1,11)
                        with gate.boundary(step):
                            history=torch.stack([encoder.encode(images[i][-width:]) for i in active])
                            previous=torch.tensor([actions[i][-width:] for i in active],device='cuda')
                            lengths=torch.full((len(active),),width,device='cuda',dtype=torch.long)
                            generated=model.generate(tasks[active],history,previous,lengths,start=step)
                            grounded=model.ground(generated,history[:,-1]).argmax(-1).cpu().numpy()
                        retained=[]
                        for j,i in enumerate(active):
                            p=export_program(generated,j);plan=controllers[i].revise(p,grounded[j],horizon=min(10,32-step),beam_width=beam_width)
                            folder=out/batch[i]['episode_id'];folder.mkdir(parents=True,exist_ok=True);name=f'program_{step:02d}.npz';p.save(folder/name)
                            records[i]['programs'].append(dict(start=step,initial_state=grounded[j].tolist(),file=name,plan=plan,
                                observed_frame_indices=list(range(max(0,step-10),step+1))))
                            if plan['found']:retained.append(i)
                            else:records[i]['planning_failed']=True
                        duration=time.perf_counter()-begin
                        for i in active:records[i]['boundary_seconds']+=duration/len(active)
                        active=retained;del generated
                    if not active:break
                    requests=[]
                    for i in active:
                        begin=time.perf_counter();action,proof=controllers[i].next_action()
                        records[i]['rule_step_seconds']+=time.perf_counter()-begin;records[i]['steps'].append(proof);requests.append((i,action))
                    responses=list(pool.map(lambda p:sims[p[0]].call('step',action=p[1]),requests));retained=[]
                    for (i,action),response in zip(requests,responses):
                        images[i].append(Image.fromarray(np.asarray(response['rgb'])[...,:3].astype(np.uint8)));actions[i].append(action)
                        # Simulator metrics and collision feedback go only into reports.
                        records[i]['steps'][-1]['collision_for_report_only']=bool(response.get('feedback',{}).get('collision',False))
                        records[i]['metrics']=response.get('metrics',{})
                        if not response['done']:retained.append(i)
                    active=retained
                    if not active:break
                for i,(e,record) in enumerate(zip(batch,records)):
                    folder=out/e['episode_id'];folder.mkdir(parents=True,exist_ok=True);images[i][0].save(folder/'first.png');images[i][-1].save(folder/'last.png')
                    proven=audit(record,folder);atomic(folder/'trace.json',record)
                    row=dict(episode_id=e['episode_id'],scene=e['scene'],domain=e['domain'],distance_group=e['distance_group'],category=e['category'],
                        success=bool(record['metrics'].get('success',False)) and not record['planning_failed'],steps=len(record['steps']),
                        collisions=sum(r['collision_for_report_only'] for r in record['steps']),planning_failed=record['planning_failed'],
                        program_count=len(record['programs']),proven_plan_count=proven,boundary_seconds=record['boundary_seconds'],rule_step_seconds=record['rule_step_seconds'])
                    rows.append(row);atomic(folder/'evaluation.json',row)
                atomic(out/'status.json',dict(state='evaluating',completed=len(rows),total=80,time=time.time()))
                print(json.dumps(dict(completed=len(rows),successes=sum(r['success'] for r in rows),planning_failures=sum(r['planning_failed'] for r in rows))),flush=True)
        result=dict(**summarize(rows),seconds=time.time()-began,suite=suite,boundary_gate_violations=gate.violations,
            all_executed_actions_have_exact_subgoal_proof=True,vision_encoding='single_frame_cuda_graph')
        assert gate.violations==0
        atomic(out/'summary.json',result);atomic(out/'status.json',dict(state='complete',time=time.time(),summary=result));return result
    finally:
        gate.close()
        for sim in sims:sim.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--suite',choices=['development','fixed80'],default='development');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'3','4','5','7'}:raise RuntimeError('GPU not authorized')
    torch.set_num_threads(2);saved=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
    if saved['manifest']['architecture']!='anonymous_visual_graph_subgoal_v5':raise ValueError('Wrong architecture')
    model=SubgoalRuleModel(Config(**saved['model_config'])).cuda();model.load_state_dict(saved['model'])
    result=evaluate(model,saved['manifest'],make_bank(),a.out,a.suite);result.update(update=saved['update'],checkpoint_sha256=sha(a.checkpoint))
    atomic(Path(a.out)/'summary.json',result)


if __name__=='__main__':main()
