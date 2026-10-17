"""Execution feedback supervises anonymous symbols, never selects an action."""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .symbols import DiscreteState


def attach_execution_outcomes(folder,episodes,demos,allowed,image_root):
    folder=Path(folder);image_root=Path(image_root).resolve()
    manifest=json.loads((folder/'manifest.json').read_text())
    path=folder/'execution_outcomes.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=manifest['outcomes_sha256']:raise ValueError('Execution outcome checksum differs')
    if not set(manifest['scene_audit']).issubset(allowed):raise ValueError('Outcome split contamination')
    rows=json.loads(path.read_text());index={}
    for eid,sequence in demos.items():
        for offset,row in enumerate(sequence):
            key=(eid,row['image'])
            if key in index:raise ValueError('Duplicate demonstration image')
            index[key]=(sequence,offset)
    allowed_keys={'image','context','history','previous_actions','episode_id','observed_collision','execution_valid'}
    seen=set();attachments=[]
    for record in rows:
        if set(record)!=allowed_keys:raise ValueError('Unexpected outcome fields')
        eid=record['episode_id'];key=(eid,record['image'])
        if key in seen:raise ValueError('Duplicate execution outcome')
        seen.add(key)
        if eid not in episodes or episodes[eid]['scene'] not in allowed:
            raise ValueError('Outcome outside training split')
        if key not in index:raise ValueError('Outcome has no matching causal demonstration')
        sequence,offset=index[key];row=sequence[offset]
        if not row.get('recovery_demonstration') or not row.get('supervision_mask') or row['action']==0:
            raise ValueError('Outcome must match an executed recovery movement')
        if record['history']!=[r['image'] for r in sequence[:offset+1]] or record['previous_actions']!=[r['action'] for r in sequence[:offset]]:
            raise ValueError('Outcome causal history mismatch')
        image=(image_root/row['image']).resolve()
        if not image.is_relative_to(image_root) or Path(row['_image_root']).resolve()!=image_root:
            raise ValueError('Outcome image escaped declared root')
        if hashlib.sha256(image.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Outcome image checksum differs')
        for name in ['observed_collision','execution_valid']:
            if len(record[name])!=17 or any(type(v) is not bool for v in record[name]):
                raise ValueError('Exactly seventeen Boolean execution labels required')
        # Channel 1 denotes no measurable movement. STOP's collector placeholder
        # is excluded: it is not a counterfactual execution observation.
        labels=[[record['observed_collision'][a],not record['execution_valid'][a]] for a in range(1,17)]
        attachments.append((row,labels))
    if len(rows)!=manifest['execution_outcome_rows']:raise ValueError('Outcome row count differs')
    for row,labels in attachments:row['execution_outcome_labels']=labels
    return {'rows':len(rows),'sha256':digest,'path':str(path),'training_only':True,
            'channels':['observed_collision','no_measurable_movement'],'stop_supervised':False}


class SymbolicExecutionDecoder(nn.Module):
    """Auxiliary only; its output is never passed to an action interpreter.

    Definition-weighted pooling is invariant to simultaneous permutation of
    anonymous predicates and their codes. Only hard symbols describe the scene.
    """
    def __init__(self,dimension,actions=17):
        super().__init__();self.action=nn.Embedding(actions,dimension)
        self.network=nn.Sequential(nn.Linear(3*dimension,dimension),nn.GELU(),nn.Linear(dimension,2))

    def forward(self,definitions,state):
        if not isinstance(state,DiscreteState):raise TypeError('Execution decoder requires discrete symbols')
        state.validate()
        if state.values.ndim!=2 or definitions.ndim!=2 or definitions.shape[0]!=state.values.shape[-1]:
            raise ValueError('Expected batched hard symbols and their generated definitions')
        codes=state.values;scale=definitions.shape[0]
        latent=torch.cat([codes@definitions/scale,(1-codes)@definitions/scale],-1)
        actions=self.action.weight[1:]
        joined=torch.cat([latent[:,None].expand(-1,len(actions),-1),actions[None].expand(len(codes),-1,-1)],-1)
        return self.network(joined)


def execution_loss(logits,labels,positive_weight=None):
    if logits.shape!=labels.shape or logits.ndim!=3 or logits.shape[1:]!=(16,2):
        raise ValueError('Expected N by 16 by 2 executed-outcome labels')
    if not torch.all((labels==0)|(labels==1)):raise ValueError('Execution labels must be Boolean')
    return F.binary_cross_entropy_with_logits(logits,labels,pos_weight=positive_weight)


def rule_execution_cost(probabilities,labels):
    """Train actual proof probabilities using measured counterfactual outcomes.

    This loss exists only during learning. It never masks, renormalizes or
    substitutes runtime actions. Conditioning on movement prevents reducing the
    objective merely by assigning more probability to STOP.
    """
    if probabilities.ndim!=2 or probabilities.shape[1]!=17 or labels.shape!=(len(probabilities),16,2):
        raise ValueError('Expected rule probabilities and sixteen measured movement outcomes')
    if not torch.isfinite(probabilities).all() or (probabilities<0).any():raise ValueError('Invalid proof probabilities')
    if not torch.all((labels==0)|(labels==1)):raise ValueError('Measured labels must be Boolean')
    movements=probabilities[:,1:];mass=movements.sum(-1,keepdim=True)
    if (mass<=0).any():raise ValueError('No supported movement for execution-cost supervision')
    measured_cost=labels.sum(-1).detach()
    return ((movements/mass)*measured_cost).sum(-1).mean()
