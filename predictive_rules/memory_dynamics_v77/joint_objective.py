"""Calibrate fixed action-sequence reachability, using existing discrete rules.

There is no learned reward, value, success, or action head. The probability is
computed by enumerating every anonymous observation outcome, followed by the
existing learned STOP transition into the model-generated joint goal. Forward
memory transitions remain categorical IDs; straight-through gradients are a
training device only. Outcome labels never enter the predictor or interpreter.
"""
import torch
from torch.nn import functional as F
from model import hard
from base_objective import JointObjective as BaseObjective,configure,TRAINABLE

def balanced_brier(p,y,valid=None):
    y=y.float();valid=torch.ones_like(y) if valid is None else valid.float()
    positive=valid*y;negative=valid*(1-y)
    a=((p-y).square()*positive).sum()/positive.sum().clamp_min(1)
    b=((p-y).square()*negative).sum()/negative.sum().clamp_min(1)
    return .5*(a+b)

def goal_membership(logits,goals):
    return hard(logits).gather(-1,goals[...,None]).squeeze(-1)

def exported_probability(model,logp,observed,actions):
    # Match compile(): a learned preserving operator has exact singleton
    # support; the training log floor is not a possible extra outcome.
    preserve=model.operator_class(actions).argmax(-1)==1
    p=torch.where(preserve[...,None],observed,logp.exp())
    return p/p.sum(-1,keepdim=True)

def derived_probabilities(model,context,memory,observation,a1,a2,goal_m,goal_o):
    """Exact first/two-action probabilities under the deployed hard-M table."""
    b=len(context);device=context.device
    lm1,lp1=model.transition(context,memory,observation,a1);m1=hard(lm1)
    og=F.one_hot(goal_o,model.cfg.observations).float()
    stop1,_=model.transition(context,m1,og,torch.zeros_like(a1))
    terminal1=torch.where(a1==0,goal_membership(lm1,goal_m),goal_membership(stop1,goal_m))
    probability_row1=exported_probability(model,lp1,observation,a1)
    p1_goal=probability_row1.gather(-1,goal_o[:,None]).squeeze(-1)
    probability1=p1_goal*terminal1
    size=model.cfg.observations;eye=torch.eye(size,device=device)[None].expand(b,-1,-1)
    lm2,lp2=model.transition(context,m1[:,None].expand(-1,size,-1),eye,a2[:,None].expand(-1,size))
    stop2,_=model.transition(context,hard(lm2),og[:,None].expand(-1,size,-1),torch.zeros(b,size,device=device,dtype=torch.long))
    targets=goal_m[:,None].expand(-1,size)
    terminal2=torch.where(a2[:,None]==0,goal_membership(lm2,targets),goal_membership(stop2,targets))
    probability_row2=exported_probability(model,lp2,eye,a2[:,None].expand(-1,size))
    p2_goal=probability_row2.gather(-1,goal_o[:,None,None].expand(-1,size,1)).squeeze(-1)
    probability2=(probability_row1*p2_goal*terminal2).sum(-1)
    return probability1,probability2

class JointObjective(BaseObjective):
    def calibrated_goal(self,rows):
        model,d=self.model,self.data
        programs,t,f0,f1,f2,a1,a2,valid,_=rows.unbind(-1);context=d.context[programs]
        with torch.no_grad():
            unique,back=torch.unique(programs,return_inverse=True);hist=d.history[unique]
            lh,_=d.logits(model,hist,unique)
            ml,_,_=model.memory_sequence(hard(lh),d.previous[unique],d.z[hist])
            m0=hard(ml)[back,t]
            known=d.runtime_memory[programs,t]
            m0=torch.where((known>=0)[:,None],F.one_hot(known.clamp_min(0),32).float(),m0)
            gm,go=model.goal_logits(context,d.task[programs]);gm=gm.argmax(-1);go=go.argmax(-1)
        l0,_=d.logits(model,f0,programs)
        p1,p2=derived_probabilities(model,context,m0,hard(l0),a1,a2,gm,go)
        y1=d.y[f1];y2=d.y[f2]
        first=balanced_brier(p1,y1);second=balanced_brier(p2,y2,valid)
        loss=first+second
        positive1=y1.float();negative1=(~y1).float();positive2=y2.float()*valid;negative2=(~y2).float()*valid
        metrics=dict(calibration_loss=loss,one_brier=first,two_brier=second,
            one_positive_probability=(p1*positive1).sum()/positive1.sum().clamp_min(1),
            one_negative_probability=(p1*negative1).sum()/negative1.sum().clamp_min(1),
            two_positive_probability=(p2*positive2).sum()/positive2.sum().clamp_min(1),
            two_negative_probability=(p2*negative2).sum()/negative2.sum().clamp_min(1))
        return loss,metrics
    def forward(self,feedback_rows,transition_rows):
        base,metrics=super().forward(feedback_rows,transition_rows)
        calibrated,more=self.calibrated_goal(transition_rows)
        total=base+4*calibrated
        metrics.update({k:v.detach() for k,v in more.items()});metrics['loss']=total.detach()
        return total,metrics
