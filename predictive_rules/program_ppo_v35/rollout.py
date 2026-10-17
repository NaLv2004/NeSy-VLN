"""RGB -> anonymous ID -> generated hard rules -> verified backward proof."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from hard_interpreter import export, plan, corrected_state
from predictive_nesy.environment import FrozenVision
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from evaluate_state512_v25 import atomic
from program_policy import semi_markov_advantages


def verify(program, proof, observed, predicted):
    current = corrected_state(program, int(predicted), int(observed))
    assert proof['found'] and proof['states'][0] == current
    for index, action in enumerate(proof['actions']):
        current = int(program.next_state[current,action])
        assert current == proof['states'][index+1]
        if action == 0: assert index == len(proof['actions'])-1
    assert current == program.goal


class Collector:
    def __init__(self, model, critic, bank, manifest, out, parallel=8):
        self.model,self.critic,self.bank,self.out = model,critic,bank,Path(out)
        self.std = manifest['context_std']
        self.generation_cost = manifest['generation_cost']
        self.failure_cost = manifest['planning_failure_penalty']
        self.budget_cost = manifest['budget_failure_penalty']
        self.frontend = FrozenVision(manifest['frozen_vision'])
        self.gate = NeuralBoundaryGate([model,self.frontend.vision],interval=1)
        self.encoder = RawBoundaryGraphEncoder(self.frontend,self.gate)
        self.sims = []
        self.pool = ThreadPoolExecutor(max_workers=parallel)
        # Initialize a few at a time rather than serialize all scene workers.
        self.sims=list(self.pool.map(lambda i:bank.simulator(self.out/f'simulator_{i}.log',0,evaluation=False),range(parallel)))
        self.inverse_hook = model.inverse_discrete.register_forward_pre_hook(self.reject_inverse)
        atomic(self.out/'processes.json',dict(simulator_pids=[s.process.pid for s in self.sims]))

    @staticmethod
    def reject_inverse(module, inputs):
        raise RuntimeError('Training-only inverse head cannot select navigation actions')

    def close(self):
        self.gate.close()
        self.inverse_hook.remove()
        self.pool.shutdown(wait=True)
        for sim in self.sims: sim.close()

    @torch.no_grad()
    def collect(self, episodes, directory, stochastic, progress=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        all_records=[];decisions=[];began=time.time()
        for offset in range(0,len(episodes),len(self.sims)):
            batch=episodes[offset:offset+len(self.sims)]
            responses=list(self.pool.map(lambda p:p[0].call('reset',episode=p[1]),zip(self.sims,batch)))
            # Deliberately ignore reset.private_metrics. No geometry goes to policy/value.
            images=[Image.fromarray(np.asarray(r['rgb'])[...,:3].astype(np.uint8)) for r in responses]
            first_images=list(images)
            histories=[[] for _ in batch];raws=[[] for _ in batch];actions=[[17] for _ in batch]
            programs={};perceptions={};predicted={}
            hidden={i:torch.zeros(1,128,device='cuda') for i in range(len(batch))}
            tasks=[torch.tensor([['chair','table'].index(e['category'])],device='cuda') for e in batch]
            records=[dict(episode=e,steps=[],programs=[],segments=[],metrics={},planning_failed=False,
                          stopped=False,reward_terms=defaultdict(float)) for e in batch]
            active=list(range(len(batch)))
            for step in range(32):
                requests=[];observations={};probabilities={i:1. for i in active}
                with self.gate.boundary(step):
                    codes,raw=self.encoder.encode_joint([images[i] for i in active])
                    local_index={i:j for j,i in enumerate(active)}
                    for j,i in enumerate(active):histories[i].append(codes[j]);raws[i].append(raw[j])
                    renewing=list(active)
                    if step:
                        old={key:torch.cat([perceptions[i][key] for i in active]) for key in perceptions[active[0]]}
                        observed=self.model.observe(old,codes,raw).argmax(-1).flatten()
                        logits,new_hidden=self.model.revision.step(observed,
                            torch.tensor([predicted[i] for i in active],device='cuda'),
                            torch.tensor([actions[i][-1] for i in active],device='cuda'),torch.cat([tasks[i] for i in active]),
                            torch.tensor([step-programs[i].start for i in active],device='cuda'),
                            torch.cat([hidden[i] for i in active]),goal=torch.tensor([programs[i].goal for i in active],device='cuda'))
                        renewing=[]
                        for j,i in enumerate(active):
                            perceptions[i]={key:value[j:j+1].clone() for key,value in old.items()}
                            hidden[i]=new_hidden[j:j+1].clone();observations[i]=int(observed[j])
                            probabilities[i]=float(logits[j].sigmoid())
                            if probabilities[i]>=.5:renewing.append(i)
                    if renewing:
                        width=min(step+1,11);count=len(renewing)
                        p,decisions_batch=self.model.decision(torch.cat([tasks[i] for i in renewing]),
                            torch.stack([torch.stack(histories[i][-width:]) for i in renewing]),
                            torch.tensor([actions[i][-width:] for i in renewing],device='cuda'),
                            torch.full((count,),width,device='cuda',dtype=torch.long),start=step,
                            visual=torch.stack([torch.stack(raws[i][-width:]) for i in renewing]),std=self.std,stochastic=stochastic)
                        values=self.critic(decisions_batch['value_input'].cuda())
                        new_codes=torch.stack([codes[local_index[i]] for i in renewing])
                        new_observed=self.model.ground(p,new_codes).argmax(-1).flatten()
                        _,new_hidden=self.model.revision.step(new_observed,new_observed,
                            torch.tensor([actions[i][-1] for i in renewing],device='cuda'),torch.cat([tasks[i] for i in renewing]),
                            torch.zeros(count,device='cuda',dtype=torch.long),torch.zeros(count,128,device='cuda'),
                            goal=p['subgoal_logits'].argmax(-1).flatten())
                        for j,i in enumerate(renewing):
                            record=records[i];program=export(p,j);programs[i]=program
                            perceptions[i]={key:p[key][j:j+1].clone() for key in ['definitions','ground_memory','ground_last','ground_codes','ground_visual']}
                            observations[i]=int(new_observed[j]);predicted[i]=observations[i];hidden[i]=new_hidden[j:j+1].clone()
                            decision={key:value[j:j+1].clone() for key,value in decisions_batch.items()}
                            decision.update(value=float(values[j]),rewards=[],executed=0,generation_cost=self.generation_cost,start=step)
                            record['segments'].append(decision)
                            event=dict(start=step,goal=program.goal,digest=program.digest(),renewal_probability=probabilities[i])
                            record['programs'].append(event)
                            if not stochastic:
                                folder=directory/batch[i]['episode_id'];folder.mkdir(parents=True,exist_ok=True)
                                name=f'program_{step:02d}.npz';event['file']=name
                                np.savez_compressed(folder/name,next_state=program.next_state,confidence=program.confidence,
                                    correction=program.correction,goal=program.goal,start=program.start)
                        del p,decisions_batch
                for i in active:
                    record=records[i];observed=observations[i]
                    proof=plan(programs[i],int(predicted[i]),int(observed),horizon=min(10,32-step))
                    if not proof['found']:
                        record['planning_failed']=True;record['failed_proof']=proof
                        record['segments'][-1]['rewards'].append(self.failure_cost)
                        record['reward_terms']['no_symbolic_proof']+=self.failure_cost
                        continue
                    verify(programs[i],proof,observed,predicted[i])
                    action=int(proof['actions'][0]);predicted[i]=int(proof['states'][1])
                    record['steps'].append(dict(step=step,observed_id=int(observed),action=action,proof=proof))
                    requests.append((i,action))
                if not requests:break
                responses=list(self.pool.map(lambda pair:self.sims[pair[0]].call('step',action=pair[1]),requests))
                retained=[]
                for (i,action),response in zip(requests,responses):
                    record=records[i];actions[i].append(action)
                    images[i]=Image.fromarray(np.asarray(response['rgb'])[...,:3].astype(np.uint8))
                    record['metrics']=response.get('metrics',{})
                    terms=record['metrics'].get('reward_terms',{})
                    reward=float(response['reward'])
                    assert abs(reward-sum(terms.values()))<1e-4, 'Reward accounting mismatch'
                    segment=record['segments'][-1];segment['rewards'].append(reward);segment['executed']+=1
                    for name,value in terms.items():record['reward_terms'][name]+=float(value)
                    record['steps'][-1].update(reward=reward,collision=bool(response.get('feedback',{}).get('collision',False)))
                    record['stopped']=action==0
                    if not response['done'] and step<31:retained.append(i)
                active=retained
                if not active:break
            for i,record in enumerate(records):
                success=bool(record['metrics'].get('success',False)) and not record['planning_failed']
                if not success and len(record['steps'])==32 and not record['stopped']:
                    record['segments'][-1]['rewards'][-1]+=self.budget_cost
                    record['reward_terms']['budget_failure']+=self.budget_cost
                semi_markov_advantages(record['segments'])
                decisions.extend(record['segments'])
                record['success']=success
                record['return']=sum(sum(s['rewards'])-s['generation_cost'] for s in record['segments'])
                record['reward_terms']['program_generation']=-self.generation_cost*len(record['segments'])
                assert abs(record['return']-sum(record['reward_terms'].values()))<1e-3
                record['segments_summary']=[{k:s[k] for k in ['start','executed','value','advantage','return','rewards','generation_cost']} for s in record['segments']]
                del record['segments']
                folder=directory/f'{offset+i:03d}_{record["episode"]["episode_id"]}' if stochastic else directory/record['episode']['episode_id']
                folder.mkdir(parents=True,exist_ok=True)
                atomic(folder/'trace.json',record)
                first_images[i].save(folder/'first.jpg');images[i].save(folder/'last.jpg')
                all_records.append(record)
            snapshot=summarize(all_records)
            snapshot.update(completed=len(all_records),total=len(episodes),seconds=time.time()-began,time=time.time())
            atomic(directory/'status.json',snapshot)
            if progress:progress(snapshot)
        result=summarize(all_records);result['seconds']=time.time()-began
        atomic(directory/'summary.json',result)
        return decisions,result


def summarize(records):
    steps=sum(len(r['steps']) for r in records);programs=sum(len(r['programs']) for r in records)
    groups=defaultdict(list)
    for record in records:
        for field in ['domain','category','distance_group']:
            groups[field+':'+record['episode'][field]].append(record)
    return dict(episodes=len(records),successes=sum(r['success'] for r in records),
        success_rate=sum(r['success'] for r in records)/max(len(records),1),
        mean_return=sum(r['return'] for r in records)/max(len(records),1),
        executed_steps=steps,rules_generated=programs,executed_steps_per_program=steps/max(programs,1),
        five_step_average_requirement_met=steps>=5*programs,
        planning_failures=sum(r['planning_failed'] for r in records),
        false_stops=sum(r['stopped'] and not r['success'] for r in records),
        collision_action_rate=sum(s['collision'] for r in records for s in r['steps'])/max(steps,1),
        all_actions_independently_audited_hard_goal_proofs=True,training_only_inverse_calls=0,
        by_group={key:dict(episodes=len(value),successes=sum(r['success'] for r in value),
            success_rate=sum(r['success'] for r in value)/len(value)) for key,value in groups.items()})
