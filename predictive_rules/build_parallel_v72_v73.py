"""Prepare two independent candidates; never modify live collection sources."""
from pathlib import Path
import ast
import shutil

BASE=Path(__file__).parent
NAMES=('continued_evidence_v72','compact_history_v73')

def write(folder,name,text):
    ast.parse(text,filename=name)
    (folder/name).write_text(text,encoding='utf-8')

for name,parent in zip(NAMES,('expanded_dynamics_v71','history_transition_v69')):
    folder=BASE/name;folder.mkdir(exist_ok=True)
    for p in (BASE/parent).glob('*.py'):
        if p.name not in ('deploy.py','launch.py','supervisor.py','handoff.py','launch_handoff.py'):
            shutil.copy2(p,folder/p.name)
    data=(BASE/'expanded_dynamics_v71/joint_data.py').read_text()
    data=data.replace("old=ROOT/'diverse_history_v70'","old=ROOT/'parallel_prefix_v72'")
    data=data.replace("assert len(names)==1200","assert len(names)==288")
    data=data.replace("assert len(self.holdout)==240","assert 0<len(self.holdout)<len(names)")
    data=data.replace('episodes=1200,fit_episodes=960,holdout_episodes=240',
                      'episodes=len(names),fit_episodes=len(names)-len(self.holdout),holdout_episodes=len(self.holdout)')
    write(folder,'joint_data.py',data)

folder=BASE/NAMES[0]
s=(folder/'joint_data.py').read_text()
# Anchor to the actual 44/80 source residual, not the older pre-calibration encoder.
s=s.replace('self.device=device;self.protocol=protocol();',
    'self.device=device;self.source_linear_weight=model.linear_visual_evidence.weight.detach().clone();self.protocol=protocol();')
s=s.replace('return logits,base',
    "source=base+torch.einsum('b...w,bow->b...o',F.linear(x,self.source_linear_weight),definitions)\n        return logits,source")
write(folder,'joint_data.py',s)
s=(folder/'train_joint.py').read_text().replace('expanded_dynamics_v71',NAMES[0]).replace("=='6,7'","=='2,3'")
s=s.replace("old=ROOT/'diverse_history_v70'","old=ROOT/'parallel_prefix_v72'")
s=s.replace('on1200fresh','on288fresh').replace('960fitting and240heldout','pre-frozen disjoint fitting/heldout')
s=s.replace("updates=1000,learning_rates", "anchor='Frozen44/80source perception rather than old global encoder',updates=1000,learning_rates")
write(folder,'train_joint.py',s)

folder=BASE/NAMES[1]
write(folder,'history_model.py','''"""Compact causal two-symbol state predictor; outputs next O, never actions."""
import torch
from torch import nn
from torch.nn import functional as F
WINDOW=2
class HistoryTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.action=nn.Embedding(18,32)
        self.query=nn.Sequential(nn.LayerNorm(576),nn.Linear(576,192),nn.SiLU(),nn.Dropout(.1),nn.Linear(192,128))
        nn.init.zeros_(self.query[-1].weight);nn.init.zeros_(self.query[-1].bias)
    def forward(self,observations,previous,mask,definitions,memory_definition,candidate_action,context,base_logp,preserve):
        evidence=definitions.gather(1,observations[...,None].expand(-1,-1,128))
        x=torch.cat([evidence,self.action(previous)],-1)*mask[...,None]
        if x.shape[1]<WINDOW:x=F.pad(x,(0,0,WINDOW-x.shape[1],0))
        query=self.query(torch.cat([x[:,-WINDOW:].flatten(1),memory_definition,candidate_action],-1))
        # Limit correction size and preserve the source distribution initially.
        residual=2*torch.tanh(torch.einsum('bw,bow->bo',query,definitions)/2)
        lp=(base_logp+residual).log_softmax(-1)
        copied=F.one_hot(observations[:,-1],64).float()
        return torch.where(preserve[:,None],copied,lp.exp()).clamp_min(1e-30).log()
def history_window(observations,previous,times):
    positions=times[:,None]+torch.arange(1-WINDOW,1,device=times.device)[None]
    mask=positions>=0;indices=positions.clamp_min(0)
    return observations.gather(1,indices),previous.gather(1,indices),mask
''')
s=(folder/'history_data.py').read_text().replace('window=8','window=2')
s=s.replace("columns={k:[]", "known=data.runtime_memory\n        self.memories=torch.where(known>=0,known,self.memories)\n        columns={k:[]")
s=s.replace("memory_training_inputs='Propagated old symbolic memory after each program boundary; no per-step image correction'",
    "memory_training_inputs='Recorded deployed symbolic memory; source propagation fallback only for unrecorded positions'")
write(folder,'history_data.py',s)
s=(folder/'train_history.py').read_text().replace('history_transition_v69',NAMES[1]).replace("=='6,7'","=='4,5'")
s=s.replace('GPUs 6/7','GPUs 4/5').replace('Last8','Last2').replace('and fixed program context','without raw program context')
s=s.replace('weight_decay=1e-4','weight_decay=.01').replace('loss=nll+.01*kl','loss=nll+.2*kl')
s=s.replace("baseline['nll']-.1", "baseline['nll']-.03").replace("baseline['accuracy']+.03", "baseline['accuracy']+.01")
s=s.replace('heldoutNLL decreases>0.1 and accuracy increases>3pp','heldoutNLL decreases>0.03 and accuracy increases>1pp')
s=s.replace("batch=2048", "batch=4096").replace('if peak<8:batch=4096','if peak<8:batch=8192')
s=s.replace('fixed_perception_and_definitions=True,', 'fixed_perception_and_definitions=True,source_anchor_kl_weight=.2,maximum_residual_logit=2,raw_context_input=False,')
write(folder,'train_history.py',s)

template=(BASE/'history_transition_v69/supervisor.py').read_text()
for name,gpus,entry in zip(NAMES,('2,3','4,5'),('train_joint.py','train_history.py')):
    s=template.replace('history_transition_v69',name).replace("'6,7'",repr(gpus)).replace('train_history.py',entry)
    write(BASE/name,'supervisor.py',s)
print('Prepared independent sources:',NAMES)
