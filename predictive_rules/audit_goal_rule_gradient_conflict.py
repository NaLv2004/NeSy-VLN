"""Measure world-model versus proof-imitation gradients, without updates."""
import json,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.graph_subgoal_model import straight_through
from predictive_nesy.graph_subgoal_training import transition,proof_log_support,state_support,mean

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
checkpoint=root/'run_vocabulary_rules_v11/checkpoint_002000.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();m.load_state_dict(saved['model'])
for name,p in m.named_parameters():p.requires_grad_(name.startswith(('transitions.','subgoal.')))
parameters=list(m.transitions.parameters())
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
def gradient(loss):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,values)])
rows=[];began=time.time()
for offset in range(0,len(selected),16):
    ws=selected[offset:offset+16];d=batch(ws,'cpu');b=len(ws)
    p=m.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
    observed=straight_through(m.training_targets(p,d['targets'],d['actions']).squeeze(2))
    desired=observed[torch.arange(b),d['goal_index']];goal=straight_through(p['subgoal_logits'].squeeze(1).softmax(-1))
    goal_match=goal.argmax(-1)==desired.argmax(-1)
    current=observed[:,:-1];future=observed[:,1:];valid=d['action_valid'].float()
    teacher=torch.einsum('bts,bsak->btak',current,p['transitions']).gather(2,d['actions'][:,:,None,None].expand(-1,-1,1,m.cfg.states)).squeeze(2)
    single=-(future*teacher.clamp_min(1e-8).log()).sum(-1)
    state=observed[:,0];predictions=[];premises=[]
    for t in range(10):
        premises.append(state);pred=transition(p,state,d['actions'][:,t]);predictions.append(pred);state=straight_through(pred)
    predictions=torch.stack(predictions,1)
    multi=-(future*predictions.clamp_min(1e-8).log()).sum(-1)
    endpoint=-(desired*predictions[torch.arange(b),d['goal_index']-1].clamp_min(1e-8).log()).sum(-1)
    levels=proof_log_support(p,goal,10);ce=[]
    for t in range(10):
        logits=state_support(premises[t],levels[10-t])/(10000*11*.5)
        ce.append(F.cross_entropy(logits,d['actions'][:,t],reduction='none'))
    ce=torch.stack(ce,1)
    for group,chosen in [('generated_goal_correct',goal_match),('generated_goal_wrong',~goal_match)]:
        if not chosen.any():continue
        mask=valid*chosen[:,None]
        dynamics=2*mean(single,mask)+.5*mean(multi,mask)+.5*mean(endpoint,chosen.float())
        imitation=mean(ce,mask);gd=gradient(dynamics);gp=gradient(imitation)
        cosine=float(F.cosine_similarity(gd,gp,dim=0))
        rows.append(dict(batch=offset//16,group=group,windows=int(chosen.sum()),actions=int(mask.sum()),
            dynamics_loss=float(dynamics.detach()),proof_loss=float(imitation.detach()),
            dynamics_gradient_norm=float(gd.norm()),proof_gradient_norm=float(gp.norm()),
            gradient_cosine=cosine,opposed=cosine<0,
            proof_directional_change_of_dynamics=float(-(gd*gp).sum())))
    del p,observed,ce,predictions,levels
result=dict(training_only=True,no_optimizer_updates=True,no_controller_changes=True,checkpoint=str(checkpoint),
    samples=len(selected),groups={})
for name in ['generated_goal_correct','generated_goal_wrong']:
    rs=[r for r in rows if r['group']==name];n=sum(r['windows'] for r in rs)
    result['groups'][name]=dict(windows=n,batches=len(rs),opposed_batches=sum(r['opposed'] for r in rs),
        weighted_mean_gradient_cosine=sum(r['gradient_cosine']*r['windows'] for r in rs)/max(n,1),
        weighted_mean_proof_to_dynamics_norm_ratio=sum(r['proof_gradient_norm']/max(r['dynamics_gradient_norm'],1e-12)*r['windows'] for r in rs)/max(n,1))
result.update(rows=rows,seconds=time.time()-began)
(root/'diagnostics/goal_rule_gradient_conflict_v11_2000.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
