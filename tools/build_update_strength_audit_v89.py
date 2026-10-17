"""Build one bounded optimizer preflight from the verified saved-rollout audit."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'scripts/audit_ppo_credit_v87.py').read_text(encoding='utf-8')
source=source.replace("out=B/'ppo_credit_v87'", "out=B/'ppo_update_strength_v89'")
source=source.replace('records=[];episodes=[]','records=[];episodes=[];recorded_batches=[]')
source=source.replace('adv,targets=advantages(reward,values,valid)', '''adv,targets=advantages(reward,values,valid)
   recorded_batches.append(dict(codes=x,actions=a,tasks=task,valid=valid,advantages=adv,
     old_probabilities=replayed['probabilities'].detach(),trials=batch))''')
source=source.replace("write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),time=time.time()))\nprint(json.dumps(result,indent=2))",'')
source+='''
# This is an optimizer preflight on recorded ORIGINAL TRAINING tasks. It does
# not replace the main training model and cannot establish navigation gains.
import ast,copy,random,numpy as np
import torch.nn.functional as F
from nesy_vla.automaton_policy_gradient import policy_objective
from nesy_vla.observation_inverse import observation_inverse_logits
source=B/'perceptual_sequences_v73/tail_1e4'
metadata=read(source/'TOKENS_COMMITTED');rows=read(source/'training_rows.json')
assert hashlib.sha256((source/'training_rows.json').read_bytes()).hexdigest()==metadata['training_rows_sha256']
assert metadata==saved['manifest']['source_commit']
codes=np.load(source/'visual_tokens.npy');all_codes=torch.tensor(codes.astype(np.int64),device='cuda')
labels=torch.tensor([r['label'] for r in rows],device='cuda');supervised=torch.tensor([r['supervised'] for r in rows],device='cuda')
weight=torch.tensor([4. if i>=metadata['old_count'] else 1. for i in range(len(rows))],device='cuda')
demo_groups={}
for i,row in enumerate(rows):demo_groups.setdefault(row['episode_id'],[]).append(i)
demos=[dict(id=eid,task=['chair','table'].index(rows[indices[0]]['category']),indices=indices) for eid,indices in demo_groups.items()]
tree=ast.parse((run/'src/train.py').read_text())
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='demonstration_loss')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(run/'src/train.py'),'exec'),globals())
model.requires_grad_(True);model.train()
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.0001)
optimizer.load_state_dict(saved['optimizer'])
for group in optimizer.param_groups:group['lr']=3e-4
rng=random.Random(8900);total_actions=sum(int(b['valid'].sum()) for b in recorded_batches)
result['preflight']=dict(actor_lr=3e-4,max_epochs=8,anchor_weight=.1,mean_kl_limit=.02,epochs=[],
 qualification='One candidate only. Eight-rank actor gradients emulated by serial accumulation of all192recordedrollouts plus64SFTepisodes. No new navigation, no validation fitting, no promoted checkpoint.')
for epoch in range(8):
 started=time.time();snapshot=copy.deepcopy(model.state_dict());optimizer_snapshot=copy.deepcopy(optimizer.state_dict())
 optimizer.zero_grad(set_to_none=True);surrogate=0.;anchor_value=0.
 for b in recorded_batches:
  r=replay(model,b['codes'],b['tasks'],b['actions']);mask=b['valid']
  oldlog=b['old_probabilities'].clamp_min(1e-20).log().gather(-1,b['actions'][...,None]).squeeze(-1)
  loss,audit=policy_objective(r['probabilities'],b['actions'],oldlog,(b['advantages']-float(mean))/float(std),mask)
  weighted=loss*int(mask.sum())/total_actions;weighted.backward();surrogate+=float(weighted.detach())
  del r,loss,weighted
 for _ in range(8):
  anchor=demonstration_loss([demos[rng.randrange(len(demos))] for _ in range(8)])
  (.1*anchor/8).backward();anchor_value+=float(anchor.detach())/8;del anchor
 norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm)
 optimizer.step()
 kl_sum=0.;changed=0;clipped=0;deltas=defaultdict(list);new_objective=0.
 with torch.no_grad():
  for b in recorded_batches:
   r=replay(model,b['codes'],b['tasks'],b['actions']);p=r['probabilities'];old=b['old_probabilities'];mask=b['valid']
   oldlog=old.clamp_min(1e-20).log();newlog=p.clamp_min(1e-20).log()
   kl_sum+=float((old*(oldlog-newlog)).sum(-1)[mask].sum())
   changed+=int(((old.argmax(-1)!=p.argmax(-1))&mask).sum())
   taken_old=oldlog.gather(-1,b['actions'][...,None]).squeeze(-1);taken_new=newlog.gather(-1,b['actions'][...,None]).squeeze(-1)
   clipped+=int(((taken_new-taken_old).exp()[mask]-1).abs().gt(.2).sum())
   loss,_=policy_objective(p,b['actions'],taken_old,(b['advantages']-float(mean))/float(std),mask)
   new_objective+=float(loss)*int(mask.sum())/total_actions
   for i,t in enumerate(b['trials']):
    for j,s in enumerate(t['steps']):
     label='success_stop' if s['action']==0 and t['metrics']['success'] else 'false_stop' if s['action']==0 else None
     if label:deltas[label].append(float(taken_new[i,j]-taken_old[i,j]))
     if j and t['steps'][j-1]['collision'] and s['action']==t['steps'][j-1]['action']:
      deltas['repeat_after_collision'].append(float(taken_new[i,j]-taken_old[i,j]))
 row=dict(epoch=epoch+1,seconds=time.time()-started,pre_update_surrogate=surrogate,post_update_surrogate=new_objective,
  anchor_loss=anchor_value,gradient_norm=norm,mean_kl=max(0.,kl_sum/total_actions),changed_greedy_fraction=changed/total_actions,
  clip_fraction=clipped/total_actions,action_log_probability_change={k:dict(count=len(v),mean=sum(v)/len(v),
    fraction_decreased=sum(x<0 for x in v)/len(v)) for k,v in deltas.items()})
 row['accepted']=np.isfinite(row['mean_kl']) and row['mean_kl']<=.02
 if not row['accepted']:
  model.load_state_dict(snapshot);optimizer.load_state_dict(optimizer_snapshot)
 result['preflight']['epochs'].append(row)
 write(out/'status.json',dict(state='optimizer_preflight',pid=os.getpid(),time=time.time(),latest=row));print(json.dumps(row),flush=True)
 if not row['accepted']:break
result['finished']=time.time();result['preflight']['accepted_epochs']=sum(r['accepted'] for r in result['preflight']['epochs'])
write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),time=time.time()))
print(json.dumps(result['preflight'],indent=2))
'''
destination=root/'scripts/audit_ppo_update_strength_v89.py'
compile(source,str(destination),'exec');destination.write_text(source,encoding='utf-8')
print(destination)
