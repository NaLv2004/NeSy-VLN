"""Isolated v57 build: one additional data-driven goal representation loss."""
from pathlib import Path
import ast,shutil
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'definition_rules_v56'
for name in ('model.py','definition_rule.py','temporal_objective.py','evaluate.py','planner.py','storage.py','program_archive.py'):
    shutil.copy2(OLD/name,HERE/name)
storage=(HERE/'storage.py').read_text(encoding='utf-8').replace('run_definition_rules_v56','run_goal_equivalence_v57')
(HERE/'storage.py').write_text(storage,encoding='utf-8')
text=(OLD/'data.py').read_text(encoding='utf-8')
needle='        assert self.branch_positions\n'
assert text.count(needle)==1
text=text.replace(needle,"""        self.terminal_pools={task:[indices[-1] for indices in self.episodes.values()
            if self.rows[indices[0]]['category']==task] for task in ('chair','table')}
        assert all(len(pool)>1 for pool in self.terminal_pools.values())
"""+needle)
needle="        result['operator_targets']=self.operator_targets\n"
assert text.count(needle)==1
text=text.replace(needle,"""        # Positive pairs use only successful demonstration endpoints in the
        # original training split. No earlier frame receives a failure label.
        alternate=[]
        for b,task in enumerate(result['task']):
            pool=self.terminal_pools[('chair','table')[int(task)]]
            own=features[b,ends[b]]
            for attempt in range(100):
                frame=self.raw[rng.choice(pool)]
                if not np.array_equal(frame,own):break
            else:raise RuntimeError('No distinct successful terminal available')
            alternate.append(frame.copy())
        result['other_success']=np.stack(alternate)
"""+needle)
(HERE/'data.py').write_text(text,encoding='utf-8')
text=(OLD/'objectives.py').read_text(encoding='utf-8')
needle='    metrics.update(ratio_metrics)\n    return total,metrics'
assert text.count(needle)==1
text=text.replace(needle,"""    metrics.update(ratio_metrics)
    from goal_objective import terminal_agreement
    terminal_loss,terminal_metrics=terminal_agreement(model,context,
        observation_logits[index,end],data['other_success'],go)
    total=total+terminal_loss
    metrics.update(terminal_metrics);metrics['loss']=total.detach()
    return total,metrics""")
(HERE/'objectives.py').write_text(text,encoding='utf-8')
text=(OLD/'train.py').read_text(encoding='utf-8')
old="""        pilot=torch.load(ROOT/'diagnostic_definition_rules_v56/head.pt',map_location='cpu',weights_only=False)
        assert pilot['source']['sha256']==source_sha
        model.load_initial(previous['model'],pilot['head'])
        del pilot"""
assert old in text
text=text.replace(old,"""        assert previous['manifest']['architecture']=='definition_rules_v56'
        model.load_state_dict(previous['model'],strict=True)""")
text=text.replace("architecture='definition_rules_v56'","architecture='goal_equivalence_v57'")
text=text.replace("in ('definition_rules_v56',)","in ('goal_equivalence_v57',)")
text=text.replace('561017','571017')
text=text.replace('Pinned v52 update2000 plus CPU-trained definition-aware categorical row generator; fresh AdamW, unchanged representation/rule learning rates',
    'Best fully evaluated v56 checkpoint; fresh AdamW at unchanged representation/rule rates')
text=text.replace("pilot_head_sha256=sha(ROOT/'diagnostic_definition_rules_v56/head.pt'),", "successful_endpoint_agreement_weight=1.0,successful_endpoint_scope='Training demonstrations only; no manually assigned terminal IDs or nonterminal failure labels',")
text=text.replace('Pinned v52 update2000; new shared categorical scorer consumes current and candidate generated observation definitions. Runtime remains exported hard-state rules.',
    'Fully evaluated v56 source; add task-supervised equivalence of successful terminal images. Runtime remains unchanged exported hard-state rules.')
(HERE/'train.py').write_text(text,encoding='utf-8')
for name in ('supervisor.py','inspect_run.py'):
    text=(OLD/name).read_text(encoding='utf-8').replace('definition_rules_v56','goal_equivalence_v57').replace('29756','29757')
    (HERE/name).write_text(text,encoding='utf-8')
text=(OLD/'check_contract.py').read_text(encoding='utf-8')
text=text.replace("pilot=torch.load(ROOT/'diagnostic_definition_rules_v56/head.pt',map_location='cpu',weights_only=False)",
    "pilot=torch.load(ROOT/'diagnostic_goal_equivalence_v57/probe.pt',map_location='cpu',weights_only=False)")
text=text.replace("model=FactoredWorld(Config(**saved['model_config']));model.load_initial(saved['model'],pilot['head'])",
    "model=FactoredWorld(Config(**saved['model_config']));model.load_state_dict(pilot['model'],strict=True)")
text=text.replace("out=ROOT/'diagnostic_definition_rules_v56'", "out=ROOT/'diagnostic_goal_equivalence_v57'")
text=text.replace("loss,metrics=objective(model,data)","assert 'other_success' in data\nloss,metrics=objective(model,data)")
text=text.replace("real_optimizer_step=True", "real_optimizer_step=True,terminal_loss=float(metrics['terminal_pair_loss']),terminal_pairs=int(len(data['other_success']))")
(HERE/'check_contract.py').write_text(text,encoding='utf-8')
for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
print('Built isolated v57. Live v56 files unchanged.')
