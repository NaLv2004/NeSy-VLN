"""GPU evaluation of the SAME finite categorical Bellman recurrence.

Only immutable learned transition tables and anonymous goals enter this code.
No neural actor, heuristic, map, or extra observation is introduced.
"""
import time
import numpy as np
import torch
from planner import Program as CPUProgram,plan,audit

@torch.no_grad()
def extend_batch(programs,horizon,device=None):
    pending=[p for p in programs if len(p._levels)<=horizon]
    if not pending:return
    device=device or torch.device('cuda',torch.cuda.current_device())
    began=time.perf_counter()
    for p in pending:p.validate()
    # All programs in a batch share the same learned state/action cardinalities.
    probs=torch.as_tensor(np.stack([p._probability for p in pending]),device=device)
    nm=torch.as_tensor(np.stack([p.next_memory for p in pending]),device=device,dtype=torch.long)
    b,s,a,o=probs.shape;assert a==17
    batch=torch.arange(b,device=device)[:,None,None]
    goal=torch.tensor([p.goal for p in pending],device=device);gm=goal//o;go=goal%o
    is_goal=nm[:,:,0]==gm[:,None]
    stop=torch.where(is_goal,probs[:,:,0].gather(-1,go[:,None,None].expand(-1,s,1)).squeeze(-1),0.)
    # Sum non-goal mass directly: no cancellation from subtracting near-one.
    non_goal=torch.arange(o,device=device)[None,None,:]!=go[:,None,None]
    stop_failure=torch.where(is_goal,(probs[:,:,0]*non_goal).sum(-1),1.)
    levels=torch.zeros(horizon+1,b,s,device=device,dtype=torch.float64)
    failures=torch.ones_like(levels);policies=torch.zeros(horizon+1,b,s,device=device,dtype=torch.int16)
    rows=torch.arange(s,device=device)[None,:];bi=torch.arange(b,device=device)[:,None]
    for h in range(1,horizon+1):
        prev=levels[h-1].reshape(b,-1,o);pf=failures[h-1].reshape(b,-1,o)
        q=(probs*prev[batch,nm]).sum(-1);f=(probs*pf[batch,nm]).sum(-1)
        q[:,:,0]=stop;f[:,:,0]=stop_failure
        action=torch.where(q.max(-1).values<=.5,q.argmax(-1),f.argmin(-1))
        levels[h]=q[bi,rows,action].clamp(0,1)
        failures[h]=f[bi,rows,action].clamp(0,1);policies[h]=action.to(torch.int16)
    # One transfer of the compact derivation; CPU proof/audit stays unchanged.
    lv=levels.cpu().numpy();fl=failures.cpu().numpy();po=policies.cpu().numpy()
    for j,p in enumerate(pending):
        p._levels=[lv[h,j].copy() for h in range(horizon+1)]
        p._failures=[fl[h,j].copy() for h in range(horizon+1)]
        p._policies=[po[h,j].copy() for h in range(horizon+1)]
        p.gpu_planning_seconds=time.perf_counter()-began

class Program(CPUProgram):
    def extend(self,horizon):
        if len(self._levels)<=horizon:extend_batch([self],horizon)

