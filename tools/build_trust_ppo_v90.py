from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'scripts/train_parallel_ppo_v85.py').read_text(encoding='utf-8')
def replace(old,new,count=1):
 global s
 assert s.count(old)==count,(old,s.count(old))
 s=s.replace(old,new)
s=s.replace('parallel_ppo_v85','trust_ppo_v90')
replace('from nesy_vla.parallel_rollout import collect_parallel','from nesy_vla.parallel_rollout import collect_parallel\nfrom nesy_vla.symbolic_trust_region import UpdateSnapshot,policy_drift')
replace("value_optimizer.load_state_dict(parent_training['value_optimizer']);del parent_training", "value_optimizer.load_state_dict(parent_training['value_optimizer']);del parent_training\nfor group in optimizer.param_groups:group['lr']=1e-4")
replace("'actor_lr':3e-5,'critic_lr':1e-3,'ppo_epochs':2", "'actor_lr':1e-4,'critic_lr':1e-3,'ppo_epochs':6,'target_kl':.01,'maximum_kl':.02,'actor_lr_limits':[3e-6,3e-4]")
replace("'selection':'Continue latest committed v82 eightGPU PPO model/optimizer/critic/allrankRNG. Only parallel rollout organization changes; no new baseline or fixed80 evaluation.'", "'selection':'Warm-start complete actor/optimizer/critic/RNG from the completed v85 development-selected best; reset this stage update counter. Adaptive exact-KL budget, no new baseline or fixed80 evaluation.'")
replace('begin=parent_epoch','begin=0')
replace('(update-parent_epoch-1)*24*world','(update-1)*24*world')
replace("   for epoch in range(2):", "   accepted_epochs=0;kl_history=[]\n   for epoch in range(6):\n    snapshot=UpdateSnapshot(model,optimizer,critic,value_optimizer,rng)")
replace('    optimizer.step();value_optimizer.step()', '''    optimizer.step();value_optimizer.step()
    with torch.no_grad():
     drift=torch.zeros(3,device='cuda',dtype=torch.float64)
     for b in batches:
      updated=replay(model,b['codes'],b['tasks'],b['actions'])
      drift+=policy_drift(b['old_probabilities'],updated['probabilities'],b['actions'],b['valid'])
     dist.all_reduce(drift)
     mean_kl=float(drift[0]/drift[2]);clip_fraction=float(drift[1]/drift[2])
     accepted=bool(torch.isfinite(drift).all()) and mean_kl<=manifest['maximum_kl']
    kl_history.append({'epoch':epoch+1,'mean_kl':mean_kl if accepted or np.isfinite(mean_kl) else None,
      'clip_fraction':clip_fraction,'accepted':accepted,'actor_lr':optimizer.param_groups[0]['lr']})
    if not accepted:
     snapshot.restore()
     for group in optimizer.param_groups:group['lr']=max(3e-6,group['lr']*.5)
     del snapshot;break
    accepted_epochs+=1
    del snapshot''')
replace("    gradient_norms.append(float(norm));losses.append(float(actor_loss.detach()));value_losses.append(float(critic_loss.detach()));bc_losses.append(float(anchor.detach()))", "    gradient_norms.append(float(norm));losses.append(float(actor_loss.detach()));value_losses.append(float(critic_loss.detach()));bc_losses.append(float(anchor.detach()))\n    if mean_kl>=manifest['target_kl']:break\n   if accepted_epochs==6 and mean_kl<manifest['target_kl']*.25:\n    for group in optimizer.param_groups:group['lr']=min(3e-4,group['lr']*1.2)")
replace("record={'collection_timing':collection", "record={'accepted_optimizer_epochs':accepted_epochs,'trust_region':kl_history,'next_actor_lr':optimizer.param_groups[0]['lr'],'collection_timing':collection")
assert 'begin=0' in s and 'range(6)' in s and 'maximum_kl' in s
compile(s,'train_trust_ppo_v90.py','exec')
(root/'scripts/train_trust_ppo_v90.py').write_text(s,encoding='utf-8')
print('Prepared one adaptive PPO continuation; no live training files modified.')
