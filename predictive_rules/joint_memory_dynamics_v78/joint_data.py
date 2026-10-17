"""Measured RGB/action outcomes restricted to the new user-selected curriculum."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import ROOT,sha
from model import hard
from curriculum import protocol

class JointData:
    @torch.no_grad()
    def __init__(self,model,holdout,device):
        self.device=device;self.protocol=protocol();old=ROOT/'diverse_history_v70'
        collection=json.loads((old/'manifest.json').read_text())
        assert json.loads((old/'status.json').read_text())['state']=='complete'
        names=[e['episode_id'] for e in collection['episodes']];assert len(names)==1200
        self.holdout=set(collection['holdout_ids']);assert len(self.holdout)==240
        xs=[];ys=[];parts=[];histories=[];previous=[];tasks=[];lengths=[];episode_offsets=[];original=[];extra=[];runtime_memories=[]
        for i,name in enumerate(names):
            p=old/'episodes'/(name+'.npz');meta=json.loads(p.with_suffix('.json').read_text())
            assert meta['domain']=='normal' and meta['distance_group'] in ('near','medium') and sha(p)==meta['sha256']
            with np.load(p,allow_pickle=False) as d:
                x=d['features'].copy();actions=d['actions'].copy();y=d['stop_success'].copy();boundaries=d['boundaries'].copy()
                extra.append(dict(features=d['future_features'].copy(),stop_success=d['future_stop_success'].copy(),
                    root_step=d['root_step'].copy(),first_action=d['first_action'].copy(),second_action=d['second_action'].copy()))
            offset=sum(len(v) for v in xs);episode_offsets.append(offset)
            xs.append(x);ys.extend(y.tolist());parts.extend([name in self.holdout]*len(x))
            history=np.full(34,offset+len(x)-1,np.int64);history[:len(x)]=np.arange(offset,offset+len(x));histories.append(history)
            prev=np.full(34,17,np.int64);prev[1:len(actions)+1]=actions;previous.append(prev)
            tasks.append(['chair','table'].index(meta['category']));lengths.append(len(x))
            original.append((actions,boundaries,y))
            deployed=np.full(34,-1,np.int64)
            for step in meta['steps']:deployed[int(step['step'])]=int(step['memory'])
            # A failed derivation has no executed step record. Unless rules
            # renewed at that position, its memory is the previous proof's M'.
            last=len(actions)
            if meta['steps'] and last not in boundaries:
                deployed[last]=int(meta['steps'][-1]['proof']['next_memory'])
            runtime_memories.append(torch.tensor(deployed,device=device))
        # Append measured counterfactual images only after indexing original histories.
        extra_offsets=[]
        for name,d in zip(names,extra):
            extra_offsets.append(sum(len(v) for v in xs));xs.append(d['features'])
            ys.extend(d['stop_success'].tolist());parts.extend([name in self.holdout]*len(d['features']))
        self.x=torch.from_numpy(np.concatenate(xs)).to(device);del xs
        self.y=torch.tensor(ys,device=device);self.frame_holdout=torch.tensor(parts,device=device)
        hist=torch.tensor(np.stack(histories),device=device);prev=torch.tensor(np.stack(previous),device=device);task=torch.tensor(tasks,device=device)
        self.z=[];global_logits=[]
        for ids in torch.arange(len(self.x),device=device).split(1024):
            logits,z,_=model.observation_encoding(self.x[ids]);self.z.append(z);global_logits.append(logits)
        self.z=torch.cat(self.z);global_o=hard(torch.cat(global_logits))
        contexts=torch.cat([model.context_sequence(self.x[hist[ids]],global_o[hist[ids]],prev[ids],task[ids])
                            for ids in torch.arange(len(names),device=device).split(32)])
        context_rows=[];program_hist=[];program_prev=[];program_task=[];program_episode=[];feedback=[];pairs=[]
        program_lookup={}
        for i,name in enumerate(names):
            actions,boundaries,y=original[i];offset=episode_offsets[i]
            for boundary in boundaries:
                key=(i,int(boundary));program_lookup[key]=len(context_rows);context_rows.append(contexts[i,int(boundary)])
                program_hist.append(hist[i]);program_prev.append(prev[i]);program_task.append(task[i]);program_episode.append(i)
            def program(t):return program_lookup[i,int(boundaries[boundaries<=t][-1])]
            n=len(actions)+(not len(actions) or actions[-1]!=0)
            for t in range(n):feedback.append((offset+t,program(t)))
            for t,a in enumerate(actions):
                valid=t+1<len(actions) and a!=0
                pairs.append((program(t),t,offset+t,offset+t+1,offset+t+2 if valid else offset+t+1,int(a),int(actions[t+1]) if valid else 1,int(valid),0))
            d=extra[i];base=extra_offsets[i];lookup={(int(t),int(a)):j for j,(t,a,b) in enumerate(zip(d['root_step'],d['first_action'],d['second_action'])) if b==-1}
            for j,t in enumerate(d['root_step']):feedback.append((base+j,program(int(t))))
            for j,(t,a,b) in enumerate(zip(d['root_step'],d['first_action'],d['second_action'])):
                t,a,b=int(t),int(a),int(b)
                if b!=-1:
                    first=lookup[t,a];pairs.append((program(t),t,offset+t,base+first,base+j,a,b,1,1))
                elif a==0:pairs.append((program(t),t,offset+t,base+j,base+j,a,1,0,1))
        self.context=torch.stack(context_rows);self.history=torch.stack(program_hist);self.previous=torch.stack(program_prev)
        self.task=torch.stack(program_task);self.program_episode=torch.tensor(program_episode,device=device)
        self.runtime_memory=torch.stack(runtime_memories)[self.program_episode]
        self.program_holdout=torch.tensor([names[i] in self.holdout for i in program_episode],device=device)
        self.definitions=F.normalize(model.observation_definitions[None]+model.observation_definition_delta(self.context),dim=-1)
        self.goal=model.goal_logits(self.context,self.task)[1].argmax(-1)
        self.feedback=torch.tensor(feedback,device=device);self.pairs=torch.tensor(pairs,device=device)
        self.parts={}
        for part,want in [('fit',False),('holdout',True)]:
            f=torch.where(self.program_holdout[self.feedback[:,1]]==want)[0]
            p=torch.where(self.program_holdout[self.pairs[:,0]]==want)[0]
            self.parts[part]=dict(feedback=f,pairs=p,positive=f[self.y[self.feedback[f,0]]],negative=f[~self.y[self.feedback[f,0]]])
            assert len(self.parts[part]['positive']) and len(self.parts[part]['negative'])
        # Preserve the source checkpoint's normalization: refitting it would
        # change its predictions even before the first optimizer update.
        self.summary=dict(episodes=1200,fit_episodes=960,holdout_episodes=240,frames=len(self.x),programs=len(self.context),
            pairs=len(self.pairs),feedback_positions=len(self.feedback),curriculum_hash=self.protocol['development_sha256'],
            counts={k:{q:len(v) for q,v in d.items()} for k,d in self.parts.items()},
            all_normal_near_medium=True,heldout_episode_ids=sorted(self.holdout),finalfixed80_used=False,
            source_collection_sha256=sha(old/'manifest.json'),source_normalization_preserved=True,
            runtime_memory_inputs='Recorded executed symbolic memory at each training root; grounding fallback only at otherwise-unrecorded rule-renewal failures')

    def logits(self,model,frames,programs):
        definitions=self.definitions[programs]
        base=12*torch.einsum('b...w,bow->b...o',self.z[frames],definitions)
        x=(self.x[frames].float().flatten(-2)-model.evidence_mean)/model.evidence_scale
        logits=base+torch.einsum('b...w,bow->b...o',model.linear_visual_evidence(x),definitions)
        return logits,base

    def sample(self,rng,feedback_size,pair_size):
        d=self.parts['fit']
        draw=lambda pool,n:pool[torch.randint(len(pool),(n,),device=self.device,generator=rng)]
        f=torch.cat([draw(d['positive'],feedback_size//2),draw(d['negative'],feedback_size-feedback_size//2)])
        # Equal mass for recorded trajectory transitions and measured alternative actions.
        p=d['pairs'];old=p[self.pairs[p,8]==0];new=p[self.pairs[p,8]==1]
        return self.feedback[f],self.pairs[torch.cat([draw(old,pair_size//2),draw(new,pair_size-pair_size//2)])]
