"""Original full dev80. Neural perception/program generation, integer planning."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from model import hard
from planner import Program,plan,audit
from program_archive import snapshot,save_recipe
from predictive_nesy.environment import FrozenVision,make_bank
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate

def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

@torch.no_grad()
def evaluate(model,manifest,out,rank=0,world_size=1,workers=4,episodes_override=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model.eval();c=model.cfg;device=next(model.parameters()).device
    archive=snapshot(model,out,rank)
    bank=make_bank();assert len(bank.dev)==80
    assert episodes_override is not None,'Training-only controlled experiment requires an explicit manifest'
    assert set(e['scene'] for e in episodes_override)<=set(manifest['data']['train_layouts'])
    assert not set(e['episode_id'] for e in episodes_override)&{e['episode_id'] for e in bank.dev}
    episodes=episodes_override[rank::world_size]
    frontend=FrozenVision(manifest['frozen_vision'])
    gate=NeuralBoundaryGate([model,frontend.vision],interval=1);encoder=RawBoundaryGraphEncoder(frontend,gate)
    inverse_calls=[0]
    def reject_inverse(*_):
        inverse_calls[0]+=1;raise RuntimeError('Auxiliary inverse head is forbidden during navigation')
    inverse_handle=model.inverse.register_forward_pre_hook(reject_inverse)
    grounding_allowed=[False]
    def reject_grounding_between_revisions(*_):
        if not grounding_allowed[0]:raise RuntimeError('Actual observation attempted to overwrite ongoing memory outside rule renewal')
    grounding_handle=model.memory.register_forward_pre_hook(reject_grounding_between_revisions)
    rows=[];sims=[];began=time.time()
    try:
        sims=[bank.simulator(out/f'sim_rank{rank}_{i}.log',int(os.environ.get('LOCAL_RANK','0')),evaluation=False) for i in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0,len(episodes),workers):
                batch=episodes[offset:offset+workers];n=len(batch)
                responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(sims,batch)))
                images=[[Image.fromarray(np.asarray(r['rgb'])[...,:3].astype(np.uint8))] for r in responses]
                features=[[] for _ in batch];actions=[[] for _ in batch];obs_history=[[] for _ in batch]
                programs={};memories={};predicted_obs={};ages={};records=[dict(episode=e,programs=[],steps=[],metrics={}) for e in batch]
                active=list(range(n))
                for step in range(32):
                    requests=[]
                    for i in active:
                        with gate.boundary(step):
                            _,raw=encoder.encode_joint([images[i][-1]])
                            features[i].append(raw[0]);observed=int(model.observe(raw).argmax(-1))
                            obs_history[i].append(observed)
                            renew=step==0;probability=1.
                            if step:
                                memory=F.one_hot(torch.tensor([memories[i]],device=device),c.memories).float()
                                obs=F.one_hot(torch.tensor([observed],device=device),c.observations).float()
                                predicted=F.one_hot(torch.tensor([predicted_obs[i]],device=device),c.observations).float()
                                probability=float(model.renewal_logits(memory,obs,predicted,torch.tensor([ages[i]],device=device)).sigmoid())
                                # At least five actual executed actions between
                                # revisions; then learned evidence decides.
                                renew=ages[i]>=5 and probability>=.5
                            if renew:
                                history=torch.stack(features[i])[None]
                                previous=torch.tensor([[17]+actions[i]],device=device)
                                obs=F.one_hot(torch.tensor([obs_history[i]],device=device),c.observations).float()
                                task=torch.tensor([['chair','table'].index(batch[i]['category'])],device=device)
                                grounding_allowed[0]=True
                                try:
                                    _,evidence,_=model.observation_encoding(history)
                                    memory_logits,_,_=model.memory_sequence(obs,previous,evidence)
                                finally:grounding_allowed[0]=False
                                memories[i]=int(memory_logits[0,-1].argmax(-1))
                                context=model.context_sequence(history,obs,previous,task)[:,-1]
                                exported=model.compile(context,task);program=Program(**exported,observations=c.observations).validate()
                                programs[i]=program;ages[i]=0
                                folder=out/batch[i]['episode_id'];folder.mkdir(parents=True,exist_ok=True)
                                filename=f'recipe_{step:02d}.npz'
                                save_recipe(folder/filename,context,task,program,archive)
                                gm,go=divmod(program.goal,c.observations)
                                terminal_preimage=bool(np.any((program.next_memory[:,0]==gm)&(program.outcome_probability[:,0,go]>0)))
                                records[i]['programs'].append(dict(start=step,file=filename,storage=archive['format'],model_archive=archive,
                                    goal=program.goal,digest=program.digest(),terminal_preimage=terminal_preimage,arithmetic='normalized_float64_dual_success_failure_v50'))
                        proof=plan(programs[i],memories[i],observed,horizon=32-step)
                        if not proof['found']:
                            records[i]['planning_failed']=True;records[i]['failed_proof']=proof;continue
                        audit(programs[i],proof)
                        action=int(proof['actions'][0])
                        successor=int(proof['next_memory'])*c.observations+int(proof['predicted_observation'])
                        records[i]['steps'].append(dict(step=step,memory=memories[i],observed=observed,action=action,
                            proof=proof,renewed=renew,renewal_probability=probability,
                            predicted_memory=successor//c.observations,predicted_observation=successor%c.observations))
                        memories[i]=successor//c.observations;predicted_obs[i]=successor%c.observations;ages[i]+=1
                        requests.append((i,action))
                    if not requests:break
                    responses=list(pool.map(lambda pair:sims[pair[0]].call('step',action=pair[1]),requests))
                    active=[]
                    for (i,action),response in zip(requests,responses):
                        actions[i].append(action);images[i].append(Image.fromarray(np.asarray(response['rgb'])[...,:3].astype(np.uint8)))
                        records[i]['metrics']=response.get('metrics',{})
                        records[i]['steps'][-1]['collision_for_report_only']=bool(response.get('feedback',{}).get('collision',False))
                        if not response['done']:active.append(i)
                    if not active:break
                for i,record in enumerate(records):
                    folder=out/batch[i]['episode_id'];folder.mkdir(parents=True,exist_ok=True)
                    atomic(folder/'trace.json',record);images[i][0].save(folder/'first.png');images[i][-1].save(folder/'last.png')
                    steps=record['steps'];pairs=list(zip(steps,steps[1:]))
                    rows.append(dict(episode_id=batch[i]['episode_id'],domain=batch[i]['domain'],distance_group=batch[i]['distance_group'],category=batch[i]['category'],
                        success=bool(record['metrics'].get('success',False)),planning_failed=record.get('planning_failed',False),
                        steps=len(steps),programs=len(record['programs']),collisions=sum(s['collision_for_report_only'] for s in steps),
                        observation_predictions=len(pairs),observation_correct=sum(a['predicted_observation']==b['observed'] for a,b in pairs),
                        stop_predicts_changed_observation=bool(steps and steps[-1]['action']==0 and steps[-1]['observed']!=steps[-1]['predicted_observation']),
                        false_stop=bool(steps and steps[-1]['action']==0 and not record['metrics'].get('success',False))))
                    changed=[(a,b) for a,b in pairs if a['observed']!=b['observed']]
                    failure=record.get('failed_proof',{})
                    rows[-1].update(initial_planning_failed=bool(failure and failure['horizon']==32),
                        last_slot_no_valid_stop=bool(failure and failure['horizon']==1),
                        programs_without_terminal_preimage=sum(not p['terminal_preimage'] for p in record['programs']),
                        adjacent_repeats=sum(a['action']==b['action'] for a,b in pairs),
                        observation_persistence_correct=sum(a['observed']==b['observed'] for a,b in pairs),
                        changed_observations=len(changed),changed_observation_correct=sum(a['predicted_observation']==b['observed'] for a,b in changed))
                atomic(out/f'rank{rank}.status.json',dict(completed=len(rows),successes=sum(r['success'] for r in rows),time=time.time()))
        atomic(out/f'rank{rank}.episodes.json',rows)
    finally:
        inverse_handle.remove();grounding_handle.remove();gate.close()
        for sim in sims:sim.close()
        del frontend,encoder
        torch.cuda.empty_cache()
    return rows


def summarize(rows):
    groups={}
    for field in ('domain','category','distance_group'):
        for value in sorted({r[field] for r in rows}):
            subset=[r for r in rows if r[field]==value]
            groups[field+':'+value]=dict(episodes=len(subset),successes=sum(r['success'] for r in subset),success_rate=sum(r['success'] for r in subset)/len(subset))
    count=sum(r['observation_predictions'] for r in rows)
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows),
        planning_failures=sum(r['planning_failed'] for r in rows),collision_action_rate=sum(r['collisions'] for r in rows)/max(1,sum(r['steps'] for r in rows)),
        observation_prediction_accuracy=sum(r['observation_correct'] for r in rows)/max(1,count),observation_predictions=count,
        actual_steps_per_program=sum(r['steps'] for r in rows)/max(1,sum(r['programs'] for r in rows)),by_group=groups,
        all_actions_symbolically_derived=True,deterministic_success_guarantee=False,proof_kind='Probabilistic discrete-rule derivation with all outcomes and a separately recorded successful witness',inverse_calls_at_runtime=0,
        false_stops=sum(r['false_stop'] for r in rows),stop_predicts_changed_observation=sum(r['stop_predicts_changed_observation'] for r in rows),
        initial_planning_failures=sum(r['initial_planning_failed'] for r in rows),
        last_slot_no_valid_stop=sum(r['last_slot_no_valid_stop'] for r in rows),
        programs_without_terminal_preimage=sum(r['programs_without_terminal_preimage'] for r in rows),
        adjacent_repeat_rate=sum(r['adjacent_repeats'] for r in rows)/max(1,count),
        observation_persistence_baseline=sum(r['observation_persistence_correct'] for r in rows)/max(1,count),
        changed_observation_accuracy=sum(r['changed_observation_correct'] for r in rows)/max(1,sum(r['changed_observations'] for r in rows)))
