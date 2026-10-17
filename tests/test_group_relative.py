import copy
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from nesy_vla.group_relative import ActorSnapshot, group_advantages, group_objective, validate_groups
from nesy_vla.grouped_rollout import collect_grouped
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton


def test_relative_sign_zero_variance_padding_and_discount():
    rewards=torch.zeros(8,4);rewards[:,0]=torch.arange(8.)
    valid=torch.ones_like(rewards,dtype=torch.bool);valid[0,1:]=False;rewards[0,1:]=10000
    advantage,audit=group_advantages(rewards,valid)
    assert audit['returns']==list(range(8))
    assert advantage[0,0]<0 and advantage[-1,0]>0
    assert float(advantage[:,0].mean().abs())<1e-6
    assert advantage[0,1:].eq(0).all()
    constant=torch.ones(8,3);a,info=group_advantages(constant,torch.ones_like(constant,dtype=torch.bool))
    assert info['zero_variance'] and a.eq(0).all()
    assert abs(info['returns'][0]-(1+.99+.99**2))<1e-5


def test_grouping_rejects_different_target_or_start():
    e={'scene':'s','episode_id':'x','start':[1,2,3],'goal':'chair'}
    batch=[copy.deepcopy(e) for _ in range(8)];validate_groups(batch)
    batch[-1]['start'][0]=2
    with pytest.raises(ValueError):validate_groups(batch)


def test_policy_gradient_increases_better_trajectory_action():
    logits=torch.zeros(8,2,3,requires_grad=True);p=logits.softmax(-1)
    actions=torch.zeros(8,2,dtype=torch.long);mask=torch.ones(8,2,dtype=torch.bool)
    advantage=torch.tensor([-1.,-1.,-1.,-1.,1.,1.,1.,1.])[:,None].expand(8,2)
    old=p.detach().log()[...,0]
    loss,audit=group_objective(p,actions,old,advantage,mask,p.detach(),beta=0,entropy_weight=0)
    loss.backward()
    assert (logits.grad[:4,:,0]>0).all() and (logits.grad[4:,:,0]<0).all()
    assert audit['max_ratio_error']==0 and abs(audit['reference_kl'])<1e-6


def test_rollout_equal_weight_and_reference_detached():
    logits=torch.zeros(8,4,3,requires_grad=True);p=logits.softmax(-1)
    ref=torch.full_like(p,1/3,requires_grad=True);mask=torch.ones(8,4,dtype=torch.bool);mask[0,1:]=False
    actions=torch.zeros(8,4,dtype=torch.long)
    loss,_=group_objective(p,actions,p.detach().log()[...,0],torch.ones(8,4),mask,ref,entropy_weight=0)
    loss.backward()
    assert torch.allclose(logits.grad[0,0],logits.grad[1].sum(0),atol=1e-7)
    assert logits.grad[0,1:].eq(0).all() and ref.grad is None


def test_actor_snapshot_restores_optimizer_and_rng():
    model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters());rng=random.Random(98)
    before=copy.deepcopy(model.state_dict());snap=ActorSnapshot(model,opt,rng)
    model(torch.ones(1,3)).sum().backward();opt.step();rng.random();torch.rand(2)
    snap.restore()
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert not opt.state and rng.getstate()==snap.python


def test_actual_grouped_collector_has_eight_identical_starts_and_exact_replay():
    torch.set_num_threads(1);torch.manual_seed(98)
    m=ProbabilisticTokenAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True)
    class Sim:
        def __init__(self,i):self.i=i;self.t=0
        def call(self,operation,**kw):
            self.t=0 if operation=='reset' else self.t+1
            return {'rgb':np.zeros((4,4,3),dtype=np.uint8),'done':self.t>=11+self.i%2,
                    'reward':float(self.i%8),'metrics':{'success':True},'feedback':{'collision':False}}
    batch=[{'category':['chair','table'][i//8%2],'episode_id':str(i//8),'scene':str(i//8)} for i in range(24)]
    with ThreadPoolExecutor(max_workers=24) as pool:
        batches,timing=collect_grouped(batch,pool,model=m,sims=[Sim(i) for i in range(24)],
            encode=lambda im:torch.zeros(6,dtype=torch.long),cfg={},task_names=['chair','table'])
    assert len(batches)==3
    for b in batches:
        assert b['initial_replay_error']==0 and b['proof_error']==0
        assert b['group_audit']['initial_images_identical']
        assert b['group_audit']['return_std']>0 and not b['group_audit']['zero_variance']
        assert 'returns' not in b and b['valid'].sum(1).tolist()==[11,12]*4
        assert len({t['episode']['episode_id'] for t in b['trials']})==1
