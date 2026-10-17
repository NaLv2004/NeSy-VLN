"""Visual abstraction and discrete dynamics see the same measured outcomes."""
import torch
from torch import nn
from torch.nn import functional as F
from model import hard

TRAINABLE=('linear_visual_evidence.','definition_rule.')
def configure(model):
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(TRAINABLE))
    model.detach_memory_carrier=False
def nll(logp,target):return -logp.gather(-1,target[...,None]).squeeze(-1)

class JointObjective(nn.Module):
    def __init__(self,model,data):super().__init__();self.model=model;self.data=data
    def feedback(self,rows):
        model,d=self.model,self.data;frames,programs=rows.unbind(-1)
        logits,base=d.logits(model,frames,programs);goal=d.goal[programs];y=d.y[frames]
        logp=logits.log_softmax(-1).gather(-1,goal[:,None]).squeeze(-1)
        p=logp.exp().clamp_max(1-1e-6);loss=torch.where(y,-logp,-torch.log1p(-p))
        anchor=(~y)&(base.argmax(-1)!=goal)
        kl=F.kl_div(logits.log_softmax(-1),base.softmax(-1),reduction='none').sum(-1)
        anchor_loss=(kl*anchor).sum()/anchor.sum().clamp_min(1)
        residual=(logits-base).square().mean();prediction=logits.argmax(-1)==goal
        return loss.mean()+.25*anchor_loss+.001*residual,dict(feedback_loss=loss.mean(),anchor_kl=anchor_loss,
            positive=y.sum(),negative=(~y).sum(),tp=(prediction&y).sum(),fp=(prediction&~y).sum(),
            positive_nll=(-logp*y).sum(),negative_nll=(-torch.log1p(-p)*(~y)).sum(),
            anchored=anchor.sum(),code_preserved=((logits.argmax(-1)==base.argmax(-1))&anchor).sum())

    def transitions(self,rows):
        model,d=self.model,self.data
        programs,t,f0,f1,f2,a1,a2,valid,kind=rows.unbind(-1);b=len(rows);idx=torch.arange(b,device=rows.device)
        context=d.context[programs]
        # Only the training label encoder sees these recorded histories/futures.
        with torch.no_grad():
            unique,back=torch.unique(programs,return_inverse=True)
            hist=d.history[unique];lh,_=d.logits(model,hist,unique)
            ml,_,_=model.memory_sequence(hard(lh),d.previous[unique],d.z[hist])
            root=hard(ml)[back,t]
            known=d.runtime_memory[programs,t]
            root=torch.where((known>=0)[:,None],F.one_hot(known.clamp_min(0),32).float(),root)
        l0,_=d.logits(model,f0,programs);l1,_=d.logits(model,f1,programs);l2,_=d.logits(model,f2,programs)
        o0=hard(l0);o1=hard(l1);o2=hard(l2)
        with torch.no_grad():
            target_m1,_=model.memory_step(root,o1.detach(),a1,d.z[f1])
            target_m2,_=model.memory_step(hard(target_m1),o2.detach(),a2,d.z[f2])
        lm1,lp1=model.transition(context,root,o0,a1)
        first_o=nll(lp1,o1.detach().argmax(-1));first_m=nll(lm1.log_softmax(-1),target_m1.argmax(-1))
        lm2,lp2=model.transition(context,hard(lm1),o1,a2)
        conditional_o=nll(lp2,o2.detach().argmax(-1));conditional_m=nll(lm2.log_softmax(-1),target_m2.argmax(-1))
        eye=torch.eye(64,device=rows.device)[None].expand(b,-1,-1)
        lm_open,lp_open=model.transition(context,hard(lm1)[:,None].expand(-1,64,-1),eye,a2[:,None].expand(-1,64))
        marginal=torch.logsumexp(lp1[:,:,None]+lp_open,dim=1)
        open_o=nll(marginal,o2.detach().argmax(-1))
        v=valid.float();second=(.5*(conditional_o+conditional_m)+open_o)*v
        loss=(first_o+first_m).mean()+second.sum()/v.sum().clamp_min(1)
        with torch.no_grad():
            changed=o0.argmax(-1)!=o1.argmax(-1);hit=lp1.argmax(-1)==o1.argmax(-1)
            metrics=dict(transition_loss=loss,first_o_nll=first_o.mean(),first_m_nll=first_m.mean(),
                open_o_nll=(open_o*v).sum()/v.sum().clamp_min(1),one_count=b,one_hits=hit.sum(),
                one_m_hits=(lm1.argmax(-1)==target_m1.argmax(-1)).sum(),changed_count=changed.sum(),changed_hits=(hit&changed).sum(),
                two_count=v.sum(),two_hits=((marginal.argmax(-1)==o2.argmax(-1))*v).sum())
        return loss,metrics

    def forward(self,feedback_rows,transition_rows):
        feedback,fm=self.feedback(feedback_rows);transition,tm=self.transitions(transition_rows)
        loss=feedback+transition
        return loss,{k:torch.as_tensor(v,device=loss.device).detach().float() for k,v in dict(fm,**tm,loss=loss).items()}
