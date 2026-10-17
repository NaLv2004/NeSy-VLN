"""Reconstruct a completed PPO update and separate actor/value gradients."""
import gc,json,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'token_revision_rl_v45';sys.path.insert(0,str(b/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic
from nesy_vla.token_rule_revision import TokenFacts,TokenProgramReviser,convert_v43_trees,PackedRuleProgram
torch.set_num_threads(2);assert read(out/'status.json')['state']=='complete'
schema=(64,)*49+(65,)*50;source=b/'spatial_automaton_v43';raw=read(source/'programs.json');tasks=sorted(raw);trees={k:PackedRuleProgram(convert_v43_trees(t,schema),schema) for k,t in raw.items()};del raw;gc.collect()
paths=list((out/'latest').glob('checkpoint.e39.*.pt'));assert len(paths)==1;checkpoint=torch.load(paths[0],map_location='cpu',weights_only=False);assert checkpoint['update']==39
model=TokenProgramReviser(schema,len(tasks),len(trees[tasks[0]].roots),detach_value_features=True);model.load_state_dict(checkpoint['model']);del checkpoint
records=[read(p) for p in sorted((out/'rollouts/step_000040').glob('*.json'))];assert len(records)==16
allrows=[r for item in records for r in item['steps']];adv=torch.tensor([r['advantage'] for r in allrows]);mean=adv.mean().item();std=adv.std(unbiased=False).clamp_min(1e-6).item();actor=[];critic=[];replay=0;weight_l1=[];maxweight=[]
for item in records:
 for chunk in item['programs']:
  rows=[r for r in item['steps'] if r['program_start']==chunk['start']];revision,value=model.generate(tasks.index(item['episode']['category']),TokenFacts(torch.tensor(chunk['history']),schema),chunk['start'])
  assert torch.allclose(revision.weights,torch.tensor(chunk['weights']),atol=1e-7)
  evidence,_=trees[item['episode']['category']].evidence(TokenFacts(torch.tensor([r['symbols'] for r in rows]),schema));p=torch.einsum('r,sra->sa',revision.weights.double(),torch.from_numpy(evidence));p=p/p.sum(-1,keepdim=True)
  actions=torch.tensor([r['action'] for r in rows]);old=torch.tensor([r['old_logp'] for r in rows],dtype=torch.float64);logp=p[torch.arange(len(rows)),actions].log();ratio=(logp-old).exp();replay=max(replay,(ratio.detach()-1).abs().max().item())
  advantage=torch.tensor([(r['advantage']-mean)/std for r in rows]);returns=torch.tensor([r['return'] for r in rows]);policy=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean();entropy=-(p*p.clamp_min(1e-30).log()).sum(-1).mean();kl=(revision.weights*(revision.weights.clamp_min(1e-30)*len(revision.weights)).log()).sum();factor=len(rows)/len(allrows)
  actor.append((policy-.01*entropy+.005*kl)*factor);critic.append(.5*F.smooth_l1_loss(value.expand_as(returns),returns)*factor)
  weight_l1.append(float((revision.weights-1/len(revision.weights)).abs().sum().detach()));maxweight.append(float(revision.weights.max().detach()))
assert replay<1e-6
named=list(model.named_parameters());params=[p for n,p in named];ag=torch.autograd.grad(torch.stack(actor).sum(),params,retain_graph=True,allow_unused=True);cg=torch.autograd.grad(torch.stack(critic).sum(),params,allow_unused=True)
def vector(grads,select):return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for (n,p),g in zip(named,grads) if select(n)])
parts={}
for name,select in [('all',lambda n:True),('program_head',lambda n:n.startswith('program.')),('shared_backbone',lambda n:not n.startswith(('program.','critic.'))),('critic_head',lambda n:n.startswith('critic.'))]:
 a=vector(ag,select);c=vector(cg,select);parts[name]={'actor_norm':float(a.norm()),'critic_norm':float(c.norm()),'total_norm':float((a+c).norm()),'cosine':float(F.cosine_similarity(a[None],c[None],dim=-1)[0]) if a.norm()>0 and c.norm()>0 else None}
summary={s.name:read(s/'summary.json') for s in (out/'evaluations').glob('step_*') if (s/'summary.json').exists()}
changed=0;steps=0
for path in (out/'evaluations/step_000040').glob('*/trace.json'):
 r=read(path);ev,_=trees[r['episode']['category']].evidence(TokenFacts(torch.tensor([s['symbols'] for s in r['steps']]),schema));baseline=ev.mean(1).argmax(1);changed+=int(np.sum(baseline!=np.array([s['action'] for s in r['steps']])));steps+=len(baseline)
assert parts['shared_backbone']['critic_norm']==0 and parts['program_head']['critic_norm']==0 and parts['program_head']['actor_norm']>0
result={'time':time.time(),'detach_value_features':True,'checkpoint_replayed':str(paths[0]),'checkpoint_sha256':sha(paths[0]),'authority':'Previous atomic-save payload validated by exact update40 rollout likelihood/weights; this is not an independent COMMITTED marker.','source_update':39,'rollout_update':40,'replay_max_error':replay,'gradient_components':parts,'rule_weights_l1_from_uniform_mean':float(np.mean(weight_l1)),'max_rule_weight_mean':float(np.mean(maxweight)),'uniform_rule_weight':1/model.trees,'final_development_actions_changed_by_reviser_at_same_symbols':changed,'final_development_actions':steps,'development':summary,'fixed80_used':False,'qualification':'Gradient decomposition is one completed training batch, not evidence of causal navigation improvement.'}
atomic(out/'detached_value_gradient_audit_v50.json',result);print(json.dumps({k:v for k,v in result.items() if k!='development'},indent=2))
