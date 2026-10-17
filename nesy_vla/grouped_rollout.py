"""Concurrent simulator rollouts with unchanged canonical symbolic batches.

RGB enters the frozen quantizer only. The policy receives integer codes.
Reusing an encoding requires a byte-identical RGB image; no approximation.
"""
import hashlib
import time
import numpy as np
import torch
from PIL import Image
from .automaton_policy_gradient import replay
from .group_relative import validate_groups, group_advantages
from .reward_experiment import training_reward


def assign_simulators(batch,sims):
    """Maximize reuse without changing a single sampled task or its order."""
    by_scene={}
    for i,sim in enumerate(sims[:len(batch)]):
        scene=getattr(sim,'_loaded_training_scene',None)
        if scene is not None:by_scene.setdefault(scene,[]).append(i)
    assignment=[None]*len(batch);used=set()
    for i,e in enumerate(batch):
        available=by_scene.get(e['scene'],[])
        if available:
            j=available.pop();assignment[i]=j;used.add(j)
    free=iter(i for i in range(len(batch)) if i not in used)
    for i,j in enumerate(assignment):
        if j is None:assignment[i]=next(free)
    assert sorted(assignment)==list(range(len(batch)))
    return [sims[j] for j in assignment],len(used)


@torch.no_grad()
def collect_grouped(batch, pool, *, model, sims, encode, cfg, task_names, horizon=32):
    validate_groups(batch)
    n=len(batch)
    if n%8 or len(sims)<n:raise ValueError('Use complete groups of eight with distinct simulators')
    device=next(model.parameters()).device
    started=time.perf_counter();timing={'encode':0.,'policy':0.,'environment':0.,'replay':0.}
    stamp=time.perf_counter()
    selected_sims,reused=assign_simulators(batch,sims)
    responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(selected_sims,batch)))
    for sim,episode in zip(selected_sims,batch):sim._loaded_training_scene=episode['scene']
    timing['environment']+=time.perf_counter()-stamp
    timing['environment_reset']=timing['environment'];timing['reused_scenes']=reused
    images=[Image.fromarray(np.asarray(r['rgb'])[...,:3].astype(np.uint8)) for r in responses]
    for start in range(0,n,8):
        hashes=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images[start:start+8]]
        assert len(set(hashes))==1,('Same task yielded different initial RGB',hashes)
    trials=[{'episode':e,'group_index':i//8,'group_member':i%8,'steps':[],'programs':[]} for i,e in enumerate(batch)]
    cache=[None]*n;cache_hits=0;cache_misses=0;groups=[]
    for start in range(0,n,8):
        _,memory=model.initial_state(8,device)
        groups.append({'start':start,'memory':memory,'previous':torch.full((8,),model.actions,device=device,dtype=torch.long),
          'tasks':torch.tensor([task_names.index(e['category']) for e in batch[start:start+8]],device=device),
          'history':[],'actions':[],'valid':[],'probabilities':[],'rewards':[], 'proof_error':0.})
    active=set(range(n));model.train()
    for step in range(horizon):
        if not active:break
        commands=[];updated=[]
        for group in groups:
            start=group['start'];ids=[i for i in range(start,start+8) if i in active]
            if not ids:continue
            stamp=time.perf_counter();tokens=[];hashes=[]
            for i in range(start,start+8):
                image=images[i];fingerprint=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                hashes.append(fingerprint)
                if cache[i] is not None and cache[i][0]==fingerprint:
                    code=cache[i][1];cache_hits+=1
                else:
                    code=encode(image);cache[i]=(fingerprint,code);cache_misses+=1
                tokens.append(code)
            x=torch.stack(tokens);group['history'].append(x)
            timing['encode']+=time.perf_counter()-stamp;stamp=time.perf_counter()
            if step%10==0:
                group['program']=model.generate(group['tasks'],torch.stack(group['history'][-11:],1),step)
                for i in ids:trials[i]['programs'].append({'start':step,'coefficients':group['program'].coefficients[i-start].tolist()})
            program=group['program'];result=model.step(program,x,group['memory'],group['previous'],step)
            proof=model.proof(program,result['observed_ids'],group['memory'],group['previous'],result['state_values'],step)
            err=float((proof['probabilities']-result['probabilities']).abs().max())
            group['proof_error']=max(group['proof_error'],err);assert err==0
            action=torch.multinomial(result['probabilities'],1).squeeze(1)
            valid=torch.zeros(8,device=device,dtype=torch.bool);valid[[i-start for i in ids]]=True
            group['valid'].append(valid);group['actions'].append(action);group['probabilities'].append(result['probabilities'])
            for i in ids:
                j=i-start;a=int(action[j]);commands.append((i,a))
                trials[i]['steps'].append({'step':step,'patch_tokens':x[j].tolist(),'observed_ids':result['observed_ids'][j].tolist(),
                  'state_ids':result['state_ids'][j].tolist(),'action':a,'probabilities':result['probabilities'][j].tolist(),
                  'program_start':program.start,'image_sha256':hashes[j]})
            group['memory']=result['state_values'];group['previous']=action
            group['rewards'].append(torch.zeros(8,device=device));updated.append(group)
            timing['policy']+=time.perf_counter()-stamp
        stamp=time.perf_counter()
        responses=list(pool.map(lambda pair:selected_sims[pair[0]].call('step',action=pair[1]),commands))
        timing['environment']+=time.perf_counter()-stamp
        for (i,_),response in zip(commands,responses):
            group=groups[i//8];adjusted=training_reward(response,cfg.get('training_reward_overrides'))
            reward=float(adjusted['reward']);group['rewards'][-1][i%8]=reward
            trials[i]['steps'][-1].update(reward=reward,collision=bool(response.get('feedback',{}).get('collision',False)),
              reward_terms=adjusted.get('training_reward_audit',{}).get('training_terms',response.get('metrics',{}).get('reward_terms',{})),
              metrics=response.get('metrics',{}))
            trials[i]['metrics']=response.get('metrics',{})
            images[i]=Image.fromarray(np.asarray(response['rgb'])[...,:3].astype(np.uint8))
            if response['done']:active.remove(i)
    stamp=time.perf_counter();batches=[]
    for g in groups:
        x=torch.stack(g['history'],1);a=torch.stack(g['actions'],1);mask=torch.stack(g['valid'],1)
        assert all(int(mask[i].sum())==len(trials[g['start']+i]['steps']) for i in range(8))
        old_probabilities=torch.stack(g['probabilities'],1)
        old=old_probabilities.clamp_min(1e-20).log().gather(-1,a[...,None]).squeeze(-1)
        adv,group_audit=group_advantages(torch.stack(g['rewards'],1),mask)
        group_audit['episode_id']=trials[g['start']]['episode']['episode_id']
        group_audit['initial_images_identical']=True
        reproduced=replay(model,x,g['tasks'],a)
        now=reproduced['probabilities'].clamp_min(1e-20).log().gather(-1,a[...,None]).squeeze(-1)
        error=float(((now-old).exp()[mask]-1).abs().max());assert error<1e-6,('Likelihood replay mismatch',error)
        batches.append({'codes':x,'actions':a,'tasks':g['tasks'],'valid':mask,'old_logp':old,
          'advantages':adv,'group_audit':group_audit,'old_probabilities':old_probabilities,'trials':trials[g['start']:g['start']+8],
          'proof_error':g['proof_error'],'initial_replay_error':error})
    timing['replay']=time.perf_counter()-stamp
    timing['environment_step']=timing['environment']-timing['environment_reset']
    timing.update(total=time.perf_counter()-started,cache_hits=cache_hits,cache_misses=cache_misses,
                  simultaneous_simulators=n,canonical_policy_batch=8)
    return batches,timing
