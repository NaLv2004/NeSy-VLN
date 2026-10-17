import threading,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.automaton_policy_gradient import SymbolicValue
from nesy_vla.parallel_rollout import collect_parallel,assign_simulators


def test_concurrent_groups_terminal_masks_exact_replay_and_cache():
    torch.set_num_threads(1);torch.manual_seed(84)
    m=ProbabilisticTokenAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True)
    c=SymbolicValue(m.factors,m.states,m.tasks);lock=threading.Lock();counts={'active':0,'peak':0,'encoded':0}
    class Sim:
        def __init__(self,i):self.i=i;self.t=0
        def call(self,operation,**kw):
            with lock:counts['active']+=1;counts['peak']=max(counts['peak'],counts['active'])
            time.sleep(.002)
            with lock:counts['active']-=1
            if operation=='reset':self.t=0
            else:self.t+=1
            # Stable image allows a byte-exact cache even while the action changes.
            return {'rgb':np.full((4,4,3),self.i%8,dtype=np.uint8),'done':self.t>=11+self.i%2,
              'reward':1.,'metrics':{'success':True,'reward_terms':{'test':1.}},'feedback':{'collision':False}}
    def encode(image):
        counts['encoded']+=1
        return torch.full((6,),int(np.asarray(image)[0,0,0]),dtype=torch.long)
    episodes=[{'category':['chair','table'][i%2],'episode_id':str(i),'scene':str(i%8)} for i in range(24)]
    with ThreadPoolExecutor(max_workers=24) as pool:
        batches,timing=collect_parallel(episodes,pool,model=m,critic=c,sims=[Sim(i) for i in range(24)],encode=encode,cfg={},task_names=['chair','table'])
    assert len(batches)==3 and counts['peak']>8 and counts['encoded']==24
    assert timing['cache_hits']>200
    for b in batches:
        assert b['initial_replay_error']==0 and b['proof_error']==0
        assert b['valid'].sum(1).tolist()==[11,12]*4
        assert all(len(t['programs'])==2 for t in b['trials'])
        assert not b['codes'].is_floating_point()


def test_scene_assignment_is_permutation_and_reuses_maximum_matches():
    class Sim:pass
    sims=[Sim() for _ in range(8)]
    for s,scene in zip(sims,['A','A','B','B','C','D','E','F']):s._loaded_training_scene=scene
    episodes=[{'scene':s} for s in ['Z','B','C','A','B','A','A','F']]
    original=[dict(e) for e in episodes];assignment,hits=assign_simulators(episodes,sims)
    assert episodes==original and hits==6
    assert len(set(map(id,assignment)))==8
    assert sum(s._loaded_training_scene==e['scene'] for s,e in zip(assignment,episodes))==6
