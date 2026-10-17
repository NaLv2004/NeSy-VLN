"""Causal discrete histories from frozen v68 perception; no validation data."""
import torch
from torch.nn import functional as F
from history_model import history_window
from model import hard

class HistoryData:
    @torch.no_grad()
    def __init__(self,model,data):
        self.data=data;self.device=data.device;self.model=model
        observations=[];memories=[]
        for ids in torch.arange(len(data.context),device=self.device).split(32):
            logits,_=data.logits(model,data.history[ids],ids)
            obs=hard(logits)
            ml,_,_=model.memory_sequence(obs,data.previous[ids],data.z[data.history[ids]])
            observations.append(obs.argmax(-1));memories.append(ml.argmax(-1))
        self.observations=torch.cat(observations);self.memories=torch.cat(memories)
        # The runtime memory is propagated by the previous rules, not regrounded
        # after every action. Reconstruct this same carrier within each program.
        starts=[]
        for p in range(len(data.context)):
            times=data.pairs[data.pairs[:,0]==p,1]
            starts.append(int(times.min()) if len(times) else 34)
        starts=torch.tensor(starts,device=self.device)
        for t in range(33):
            ids=torch.where(starts<=t)[0]
            if not len(ids):continue
            lm,_=model.transition(data.context[ids],F.one_hot(self.memories[ids,t],32).float(),
                F.one_hot(self.observations[ids,t],64).float(),data.previous[ids,t+1].clamp_max(16))
            self.memories[ids,t+1]=lm.argmax(-1)
        known=data.runtime_memory
        self.memories=torch.where(known>=0,known,self.memories)
        columns={k:[] for k in ('program','obs','previous','mask','memory','action','target','base_logp','holdout','kind','changed')}
        for rows in data.pairs.split(256):
            p,t,f0,f1,f2,a1,a2,valid,kind=rows.unbind(-1)
            hist,prev,mask=history_window(self.observations[p],data.previous[p],t)
            o1=data.logits(model,f1,p)[0].argmax(-1);o2=data.logits(model,f2,p)[0].argmax(-1)
            memory=self.memories[p,t]
            lm,lp=model.transition(data.context[p],F.one_hot(memory,32).float(),F.one_hot(hist[:,-1],64).float(),a1)
            def append(pr,h,prev,mask,m,a,target,base,k):
                vals=dict(program=pr,obs=h,previous=prev,mask=mask,memory=m,action=a,target=target,base_logp=base,
                          holdout=data.program_holdout[pr],kind=k,changed=h[:,-1]!=target)
                for name,value in vals.items():columns[name].append(value)
            append(p,hist,prev,mask,memory,a1,o1,lp,kind)
            keep=valid.bool()
            h2=torch.cat([hist[:,1:],o1[:,None]],-1)
            prev2=torch.cat([prev[:,1:],a1[:,None]],-1)
            mask2=torch.cat([mask[:,1:],torch.ones(len(p),1,device=self.device,dtype=torch.bool)],-1)
            _,lp2=model.transition(data.context[p],hard(lm),F.one_hot(o1,64).float(),a2)
            append(p[keep],h2[keep],prev2[keep],mask2[keep],lm.argmax(-1)[keep],a2[keep],o2[keep],lp2[keep],kind[keep])
        self.columns={k:torch.cat(v) for k,v in columns.items()}
        self.fit=torch.where(~self.columns['holdout'])[0];self.holdout=torch.where(self.columns['holdout'])[0]
        self.original=self.fit[self.columns['kind'][self.fit]==0]
        self.branch=self.fit[self.columns['kind'][self.fit]==1]
        self.memory_definitions=model.definitions(data.context)
        self.summary=dict(rows=len(self.columns['target']),fit_rows=len(self.fit),holdout_rows=len(self.holdout),
            window=2,source_perception_frozen=True,all_normal_near_medium=True,validation_used_for_training=False,
            memory_training_inputs='Recorded deployed symbolic memory; source propagation fallback only for unrecorded positions')

    def inputs(self,ids,history=True):
        x={k:v[ids] for k,v in self.columns.items()};p=x['program'];a=x['action']
        mask=x['mask'].clone()
        if not history:mask[:,:-1]=False
        return dict(observations=x['obs'],previous=x['previous'],mask=mask,
            definitions=self.data.definitions[p],memory_definition=self.memory_definitions[p,x['memory']],
            candidate_action=self.model.action_embedding(a),context=self.data.context[p],
            base_logp=x['base_logp'],preserve=self.model.operator_class(a).argmax(-1)==1)

    def sample(self,rng,n):
        draw=lambda pool,k:pool[torch.randint(len(pool),(k,),device=self.device,generator=rng)]
        return torch.cat([draw(self.original,n//2),draw(self.branch,n-n//2)])

