"""Training-only action experiments; labels never enter the propagated state."""
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn.functional as F
from .data import read,sha
from .model import straight_through
from .model_memory import transition_all,risk_all,scalar
from .training_memory import bce


class CounterfactualBank:
    def __init__(self,directory,train_layouts):
        self.directory=Path(directory);self.manifest=read(self.directory/'manifest.json')
        if not self.manifest['training_only']:raise ValueError('Outcome labels must be training-only')
        if set(self.manifest['data']['train_layouts'])!=set(train_layouts):raise ValueError('Training split changed')
        self.allowed=set(self.manifest['episodes']);self.train_layouts=set(train_layouts)
        self.episodes={};self.by_key={};self.keys=[]
        self.refresh()

    def refresh(self):
        for path in sorted(self.directory.glob('*.npz')):
            if path.name.endswith('.tmp.npz'):continue
            if path.stem in self.episodes:continue
            if path.stem not in self.allowed:raise ValueError('Unlisted outcome trajectory')
            match=re.search(r'P25_train_(\d+)_',path.stem)
            if match is None or 'layout_'+match.group(1) not in self.train_layouts:raise ValueError('Held-out outcome trajectory')
            with np.load(path,allow_pickle=False) as source:data={name:source[name].copy() for name in source.files}
            if any(x in data for x in ['pose','map','goal_coordinates','distance']):raise ValueError('Privileged policy feature')
            n=len(data['actions'])
            if data['codes'].shape!=(n,49) or data['successors'].shape!=(n,17,49):raise ValueError('Outcome dimensions changed')
            # A branch is usable only after its simulator implementation agrees
            # with the real execution path at this particular starting state.
            self.episodes[path.stem]=data
            for t in range(n):
                if not data['probe_valid'][t]:continue
                key=(t//10*10,t%10)
                self.by_key.setdefault(key,[]).append((path.stem,t));self.keys.append(key)
        return dict(episodes=len(self.episodes),states=len(self.keys),manifest_sha256=sha(self.directory/'manifest.json'))

    def sample(self,rng,batch_size,device):
        if not self.keys:raise ValueError('No verified counterfactual labels available')
        start,offset=rng.choice(self.keys);chosen=rng.choices(self.by_key[start,offset],k=batch_size)
        rows=[]
        for name,t in chosen:
            d=self.episodes[name];begin=max(0,start-10)
            rows.append(dict(task=int(d['task']),history=d['codes'][begin:start+1],
                previous_actions=np.array([int(d['actions'][j-1]) if j else 17 for j in range(begin,start+1)]),
                prefix_codes=d['codes'][start:t+1],prefix_actions=d['actions'][start:t],
                successors=d['successors'][t],collision=d['collision'][t],
                eligible_next=d['eligible_next'][t],eligible_current=d['eligible_current'][t]))
        batch={k:torch.as_tensor(np.asarray([r[k] for r in rows]),device=device) for k in rows[0]}
        for k in ['task','history','previous_actions','prefix_codes','prefix_actions','successors']:batch[k]=batch[k].long()
        batch.update(start=start,offset=offset,episode_ids=[n for n,t in chosen])
        return batch


def propagated_predictions(model,batch):
    b=len(batch['task']);c=model.cfg
    lengths=torch.full((b,),batch['history'].shape[1],device=batch['task'].device,dtype=torch.long)
    program=model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=batch['start'])
    state=straight_through(model.ground(program,batch['prefix_codes'][:,0]))
    # At runtime this is the complete prediction path. The label encoder below
    # is separate and cannot replace any of these predicted states.
    for j in range(batch['offset']):
        state=transition_all(program,state)[torch.arange(b,device=state.device),batch['prefix_actions'][:,j]]
    predicted=transition_all(program,state)
    predicted=torch.cat([state[:,None],predicted[:,1:]],1) # STOP leaves the state unchanged.
    return program,state,predicted


def objective(model,batch):
    b=len(batch['task']);c=model.cfg
    program,state,predicted=propagated_predictions(model,batch)
    with torch.no_grad():
        memory=program['context_raw'].detach()
        for j in range(batch['offset']):
            z=model.context_read(model.embed(batch['prefix_codes'][:,j+1]))+model.action_embed(batch['prefix_actions'][:,j])
            _,h=model.history(z[:,None],memory[None]);memory=h[0]
        actions=torch.arange(c.actions,device=state.device)[None].expand(b,-1)
        z=model.context_read(model.embed(batch['successors']))+model.action_embed(actions)
        _,h=model.history(z.reshape(b*c.actions,1,-1),memory.repeat_interleave(c.actions,0)[None])
        next_memory=h[0].reshape(b,c.actions,-1);next_memory[:,0]=memory
    # Ground truth pictures are supervision only. Gradients through grounding
    # train the vocabulary; they never supply future state to the rollout.
    branch_program={'definitions':program['definitions'].repeat_interleave(c.actions,0)}
    recognized=model.ground_features(branch_program,batch['successors'].flatten(0,1),next_memory.flatten(0,1)).reshape(b,c.actions,c.factors,c.states)
    recognized=straight_through(recognized)
    current=recognized[:,0]
    losses={}
    losses['transition']=-(recognized.detach()*predicted.clamp_min(1e-8).log()).sum(-1).mean()
    losses['current_state']=-(current.detach()*state.clamp_min(1e-8).log()).sum(-1).mean()
    risk_label=batch['collision'].float()
    losses['collision']=(bce(risk_all(program,state)[:,1:],risk_label[:,1:]).mean()+bce(risk_all(program,current)[:,1:],risk_label[:,1:]).mean())/2
    future_goal=batch['eligible_next'].float()
    losses['goal']=(bce(scalar(program,predicted,'goals',model.pairs),future_goal).mean()+bce(scalar(program,recognized,'goals',model.pairs),future_goal).mean())/2
    logits=model.reconstruct(recognized.flatten(-2)).reshape(b,c.actions,c.patches,c.vocabulary)
    losses['reconstruction']=F.cross_entropy(logits.flatten(0,2),batch['successors'].flatten())
    # Calibrate the only known value target: eligibility means STOP can succeed.
    # No shortest-path value or goal distance is supplied.
    goal_value=scalar(program,predicted,'values',model.pairs)
    losses['terminal_value']=((goal_value-1).square()*future_goal).sum()/future_goal.sum().clamp_min(1)
    total=losses['transition']+.25*losses['current_state']+2*losses['collision']+2*losses['goal']+.15*losses['reconstruction']+.2*losses['terminal_value']
    metrics={'cf_'+k:float(v.detach()) for k,v in losses.items()}
    metrics.update(cf_loss=float(total.detach()),cf_collision_pred=float(risk_all(program,state)[:,1:].mean().detach()),
        cf_collision_true=float(risk_label[:,1:].mean()),cf_offset=batch['offset'])
    return total,metrics
