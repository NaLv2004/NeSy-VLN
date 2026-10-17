"""One-time build; keep the live collector's model and interpreter unchanged."""
from pathlib import Path
here=Path(__file__).parent;old=here.parent/'goal_aware_symbols_v52'
s=(old/'data.py').read_text(encoding='utf-8')
begin=s.index('        extra_provenance=None');end=s.index('        self.effect_evidence=',begin)
s=s[:begin]+'''        extra_provenance=[]
        if branch_root:
            new=Path(branch_root).resolve()
            assert new.parent==ROOT and new.name=='onpolicy_successors_v54'
            for extra in (ROOT/'onpolicy_successors_v40',new):
                info=read(extra/'manifest.json');extra_provenance.append(info)
                assert info['training_only'] and info['no_goal_or_imitation_supervision']
                assert set(info['train_layouts'])==self.layouts
                assert read(extra/'status.json')['state']=='complete'
                metas=sorted((extra/'counterfactual_onpolicy').glob('*.json'))
                assert len(metas)==len(info['episode_ids'])
                assert {read(p)['episode'] for p in metas}==set(info['episode_ids'])
                is_current=extra==new
                for meta in metas[rank::world_size]:
                    record=read(meta);self.check_layout(record['episode'])
                    assert record['training_only'] and not record['goal_supervision']
                    assert record['scene'] in self.layouts and not record['probe_labels_available_to_policy']
                    path=meta.with_suffix('.npz');assert sha(path)==record['output_sha256']
                    assert record['source_checkpoint_sha256']==info['source_commit']['sha256']
                    with np.load(path,allow_pickle=False) as archive:
                        d={k:archive[k] for k in ('features','successor_features','recorded_actions','probe_valid')}
                    n=len(d['features']);assert 1<=n<=32 and d['features'].shape==(n,49,128)
                    assert d['successor_features'].shape==(n,17,49,128)
                    assert len(d['recorded_actions']) in (n,n-1)
                    assert np.isfinite(d['features']).all() and np.isfinite(d['successor_features']).all()
                    idx=len(self.branches);self.branches.append((record,d))
                    positions=[(idx,t) for t,v in enumerate(d['probe_valid']) if v and t<32]
                    self.branch_positions.extend(positions)
                    (self.onpolicy_positions if is_current else self.old_branch_positions).extend(positions)
            assert self.onpolicy_positions
''' + s[end:]
s=s.replace("bt.append(t);bs.append(rng.randrange(t+1))", "bt.append(t)\n            bs.append(max(p['start'] for p in record['programs'] if p['start']<=t) if is_onpolicy else rng.randrange(t+1))")
s=s.replace("context_boundary_sampling='unchanged'", "context_boundary_sampling='Recorded actual renewal boundary for fresh-policy states; original random prefix for older data'")
(here/'data.py').write_text(s,encoding='utf-8')
s=(old/'objectives.py').read_text(encoding='utf-8')
s=s.replace('    endpoint_m=None;endpoint_o=None','    endpoint_m=None;endpoint_o=None\n    hybrid_memory=sm.detach();hybrid_losses=[];hybrid_o_correct=[];hybrid_m_correct=[]')
needle='        sm,ob=hard(next_m),hard(next_o)'
addition='''        # Match runtime: predicted M_t plus freshly observed O_t. Never
        # replace the memory with a grounded post-action state during rollout.
        hm,ho=model.transition(context,hybrid_memory,world_observations[index,t],action,definitions)
        hybrid_losses.append(ce(hm,tm,valid)+ce(ho,to,valid))
        hybrid_o_correct.append(accuracy(ho,to,valid));hybrid_m_correct.append(accuracy(hm,tm,valid))
        hybrid_memory=hard(hm).detach()
'''
assert needle in s;s=s.replace(needle,addition+needle)
needle='    branch=ce(pred_m,targets_m)+ce(pred_o,targets_o)'
addition='''
    # Replay the actual action prefix with current observations available but
    # no memory grounding after its renewal. Prefix is data distribution, not
    # an extra long backpropagation path. Only the scored row receives a loss.
    with torch.no_grad():
        hybrid_bm=bm[bi,bs].detach()
        for offset in range(int((bt-bs).max())):
            t=(bs+offset).clamp_max(bx.shape[1]-2)
            active=(bs+offset)<bt
            prefix_action=torch.where(active,bprevious[bi,t+1],torch.zeros_like(t))
            next_hm,_=model.transition(bc.detach(),hybrid_bm,bo[bi,t].detach(),prefix_action)
            hybrid_bm=torch.where(active[:,None],hard(next_hm),hybrid_bm)
    hybrid_bm0=hybrid_bm[:,None].expand(-1,c.actions,-1)
    hybrid_pred_m,hybrid_pred_o=model.transition(bc,hybrid_bm0,o0,a)
    hybrid_branch=ce(hybrid_pred_m,targets_m)+ce(hybrid_pred_o,targets_o)
    hybrid_replay_loss=hybrid_branch+torch.stack(hybrid_losses).mean()
'''
assert needle in s;s=s.replace(needle,needle+addition)
s=s.replace('    total=forward+branch+goal+', '    total=hybrid_replay_loss+forward+branch+goal+')
needle='    return total,metrics'
addition='''    metrics.update(hybrid_replay_loss=hybrid_replay_loss.detach(),
        hybrid_branch_loss=hybrid_branch.detach(),
        hybrid_branch_observation_accuracy=accuracy(hybrid_pred_o,targets_o),
        hybrid_branch_memory_accuracy=accuracy(hybrid_pred_m,targets_m),
        hybrid_current_memory_disagreement=(hybrid_bm.argmax(-1)!=bm[bi,bt].detach().argmax(-1)).float().mean(),
        hybrid_branch_mean_age=(bt-bs).float().mean(),
        hybrid_demo_observation_accuracy=torch.stack(hybrid_o_correct).mean(),
        hybrid_demo_memory_accuracy=torch.stack(hybrid_m_correct).mean())
'''
s=s.replace(needle,addition+needle);(here/'objectives.py').write_text(s,encoding='utf-8')
for name in ('train.py','storage.py','supervisor.py'):
    s=(old/name).read_text(encoding='utf-8').replace('goal_aware_symbols_v52','hybrid_replay_v54')
    if name=='train.py':
        s=s.replace('model.load_v48(previous[\'model\'])',"model.load_state_dict(previous['model'],strict=True)")
        s=s.replace("'v48 best500 weights initialize existing modules; new task-conditioned abstraction embedding zero-initialized; fresh AdamW'", "'v52 best2000 exact weights; fresh AdamW; same architecture and rates'")
        s=s.replace("source_model='v48 best500; no Qwen vision substitution'", "source_model='v52 best2000; current-policy factual branch aggregation plus runtime-matched predicted-memory/observed-state supervision'")
        s=s.replace("fixed_development_episodes=80,max_actions=32)","fixed_development_episodes=80,max_actions=32,\n        hybrid_replay='Teacher-grounded labels, predicted memory carrier, actual current observation; memory grounds only at sampled rule boundary; no extra neural planner',\n        fresh_onpolicy_fraction=.5,auxiliary_hybrid_loss_weight=1.0)")
    if name=='supervisor.py':
        s=s.replace('--master_port=29752','--master_port=29754')
        s=s.replace("'--updates','3000'","'--updates','2000'")
        s=s.replace("'--branch-root',str(ROOT/'onpolicy_successors_v40')", "'--branch-root',str(ROOT/'onpolicy_successors_v54')")
        s=s.replace("ROOT/'diagnostic_hybrid_replay_v54/source.pt'","ROOT/'onpolicy_successors_v54/source.pt'")
    (here/name).write_text(s,encoding='utf-8')
print('Prepared v54 runtime-matched hybrid supervision; same model and exact planner.')
