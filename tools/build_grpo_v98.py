"""Build an isolated GRPO successor from the verified PPO rollout/evaluation code."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
s = (root/'nesy_vla/parallel_rollout.py').read_text(encoding='utf-8')
def edit(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
edit('from .automaton_policy_gradient import replay, advantages',
     'from .automaton_policy_gradient import replay\nfrom .group_relative import validate_groups, group_advantages')
edit('def collect_parallel(batch, pool, *, model, critic, sims, encode, cfg, task_names, horizon=32):',
     'def collect_grouped(batch, pool, *, model, sims, encode, cfg, task_names, horizon=32):')
edit('    n=len(batch)', '    validate_groups(batch)\n    n=len(batch)')
edit("    trials=[{'episode':e,'steps':[],'programs':[]} for e in batch]", """    for start in range(0,n,8):
        hashes=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images[start:start+8]]
        assert len(set(hashes))==1,('Same task yielded different initial RGB',hashes)
    trials=[{'episode':e,'group_index':i//8,'group_member':i%8,'steps':[],'programs':[]} for i,e in enumerate(batch)]""")
edit("'values':[],", '')
edit("            group['values'].append(critic(result['state_values'],group['tasks']))\n", '')
edit(";v=torch.stack(g['values'],1)", '')
edit("        adv,returns=advantages(torch.stack(g['rewards'],1),v,mask)",
     "        adv,group_audit=group_advantages(torch.stack(g['rewards'],1),mask)\n        group_audit['episode_id']=trials[g['start']]['episode']['episode_id']\n        group_audit['initial_images_identical']=True")
edit("'advantages':adv,'returns':returns,", "'advantages':adv,'group_audit':group_audit,")
compile(s, 'grouped_rollout.py', 'exec')
(root/'nesy_vla/grouped_rollout.py').write_text(s, encoding='utf-8')

s = (root/'scripts/train_trust_ppo_v90.py').read_text(encoding='utf-8')
s = s.replace('trust_ppo_v90', 'group_grpo_v98')
edit('import argparse,datetime,', 'import argparse,copy,datetime,')
edit('from nesy_vla.automaton_policy_gradient import SymbolicValue,replay,advantages,policy_objective,configure_trainable',
     'from nesy_vla.automaton_policy_gradient import replay,configure_trainable\nfrom nesy_vla.group_relative import group_objective,ActorSnapshot')
edit('from nesy_vla.parallel_rollout import collect_parallel', 'from nesy_vla.grouped_rollout import collect_grouped')
edit('from nesy_vla.symbolic_trust_region import UpdateSnapshot,policy_drift', 'from nesy_vla.symbolic_trust_region import policy_drift')
edit(";parent_training={k:warm[k] for k in ['optimizer','critic','value_optimizer']}", '')
start = s.index('critic=SymbolicValue(')
end = s.index('manifest={', start)
s = s[:start] + """reference=copy.deepcopy(model).eval().requires_grad_(False)
optimizer=torch.optim.AdamW(trainable,lr=3e-5,weight_decay=.0001)
""" + s[end:]
edit("'actor_lr':1e-4,'critic_lr':1e-3,'ppo_epochs':6", "'actor_lr':3e-5,'grpo_epochs':2,'algorithm':'outcome_GRPO','group_size':8,'groups_per_rank':3,'global_groups_per_update':24,'reference_kl_beta':.01,'episode_equal_weighting':True")
edit("'source_development':read(out/'SOURCE.json')['development_before_parallel_change']", "'source_development':read(out/'SOURCE.json')['selected_development']")
edit("'optimizer_resume':True,'source_sft_optimizer_retained':True", "'optimizer_resume':False,'fresh_actor_optimizer_at_algorithm_switch':True,'critic_used':False")
edit("'selection':'Warm-start complete actor/optimizer/critic/RNG from the completed v85 development-selected best; reset this stage update counter. Adaptive exact-KL budget, no new baseline or fixed80 evaluation.'",
     "'selection':'Restore exact v85 PPO70 actor (43/80 development); new GRPO optimizer. G=8 same-reset trajectories, discounted shaped return normalized within group, no critic. No repeated baseline or fixed80 evaluation.'")
edit("critic.load_state_dict(saved['critic']);value_optimizer.load_state_dict(saved['value_optimizer']);begin=saved['epoch']", "begin=saved['epoch']")
edit('assert_identical_parameters(model);assert_identical_parameters(critic)', 'assert_identical_parameters(model)', 2)
edit("'model':model.state_dict(),'optimizer':optimizer.state_dict(),'critic':critic.state_dict(),\n    'value_optimizer':value_optimizer.state_dict(),", "'model':model.state_dict(),'optimizer':optimizer.state_dict(),")
# Remove the unused legacy serial collector: only critic-free grouped collection remains.
start = s.index('@torch.no_grad()\ndef collect(')
end = s.index('def trial_metric(', start)
s = s[:start] + s[end:]
start = s.index('   batch=[bank.sample(')
end = s.index('   losses=[];', start)
s = s[:start] + """   starts=[bank.sample(rng,manifest['sample_index_start']+(update-1)*3*world+rank*3+i) for i in range(3)]
   batch=[copy.deepcopy(e) for e in starts for _ in range(8)]
   assert all(e['scene'] in bank.provenance['train_layouts'] for e in batch)
   batches,collection=collect_grouped(batch,pool,model=model,sims=sims,encode=encode,cfg=cfg,task_names=tasks)
   optimization_started=time.time()
   with torch.no_grad():
    for b in batches:b['reference']=replay(reference,b['codes'],b['tasks'],b['actions'])['probabilities'].detach()
   total_actions=sum(int(b['valid'].sum()) for b in batches)
   global_actions=torch.tensor(total_actions,device='cuda');dist.all_reduce(global_actions)
   group_audits=[dict(b['group_audit'],rank=rank) for b in batches]
   all_group_audits=[r for part in gather_objects(group_audits) for r in part]
   assert len(all_group_audits)==24
""" + s[end:]
edit('losses=[];value_losses=[];', 'losses=[];reference_kls=[];')
edit('for epoch in range(6):', 'for epoch in range(2):')
edit('snapshot=UpdateSnapshot(model,optimizer,critic,value_optimizer,rng)', 'snapshot=ActorSnapshot(model,optimizer,rng)')
edit('optimizer.zero_grad(set_to_none=True);value_optimizer.zero_grad(set_to_none=True)', 'optimizer.zero_grad(set_to_none=True)')
edit('actor_parts=[];critic_parts=[]', 'actor_parts=[]')
edit("loss,audit=policy_objective(result['probabilities'],b['actions'],b['old_logp'],(b['advantages']-mean)/std,b['valid'])",
     "loss,audit=group_objective(result['probabilities'],b['actions'],b['old_logp'],b['advantages'],b['valid'],b['reference'])\n     reference_kls.append(audit['reference_kl'])")
start = s.index("     actor_parts.append(loss*")
end = s.index('    if epoch==0:assert', start)
s = s[:start] + '     actor_parts.append(loss/len(batches))\n' + s[end:]
edit("('PPO likelihood mismatch before update',replay_error)", "('GRPO likelihood mismatch before update',replay_error)")
edit('actor_loss=rank_weight*sum(actor_parts)+.1*anchor;critic_loss=rank_weight*sum(critic_parts)', 'actor_loss=sum(actor_parts)+.1*anchor')
edit('assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)', 'assert torch.isfinite(actor_loss)')
edit('    critic_loss.backward();average_gradients(critic.parameters());value_norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),5.);assert torch.isfinite(value_norm)\n', '')
edit('optimizer.step();value_optimizer.step()', 'optimizer.step()')
edit(';value_losses.append(float(critic_loss.detach()))', '')
edit("   if accepted_epochs==6 and mean_kl<manifest['target_kl']*.25:\n    for group in optimizer.param_groups:group['lr']=min(3e-4,group['lr']*1.2)\n", '')
edit("'actor_losses':losses,'value_losses':value_losses,", "'algorithm':'GRPO','group_audits':all_group_audits,'zero_variance_groups':sum(g['zero_variance'] for g in all_group_audits),'actor_losses':losses,'reference_kls':reference_kls,")
edit("'ppo_replay_max_ratio_error':replay_error", "'grpo_replay_max_ratio_error':replay_error")
edit("'ppo_clip_fractions':clip_fractions", "'grpo_clip_fractions':clip_fractions")
edit('del batches,result,actor_parts,critic_parts,anchor,actor_loss,critic_loss', 'del batches,result,actor_parts,anchor,actor_loss')
assert 'SymbolicValue' not in s and 'value_optimizer' not in s and 'critic_loss' not in s
assert 'group_advantages' not in s and 'group_objective' in s
compile(s, 'train_group_grpo_v98.py', 'exec')
(root/'scripts/train_group_grpo_v98.py').write_text(s, encoding='utf-8')
print('Built isolated critic-free same-task GRPO trainer and collector.')
