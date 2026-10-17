"""Prepare a controlled new-data refinement; this script does not launch it."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
for name in ('train.py','supervisor.py','storage.py'):
    s=(BASE/'goal_equivalence_v57'/name).read_text(encoding='utf-8')
    s=s.replace('src_goal_equivalence_v57','src_onpolicy_refresh_v60_training').replace('run_goal_equivalence_v57','run_onpolicy_refresh_v60')
    s=s.replace('diagnostic_goal_equivalence_v57/source.pt','onpolicy_successors_v60/source.pt')
    s=s.replace('onpolicy_successors_v54','onpolicy_successors_v60')
    if name=='train.py':
        s=s.replace('571017','601017')
        s=s.replace("assert previous['manifest']['architecture']=='definition_rules_v56'","assert previous['manifest']['architecture']=='goal_equivalence_v57'")
        s=s.replace("architecture='goal_equivalence_v57'","architecture='onpolicy_refresh_v60'")
        s=s.replace("in ('goal_equivalence_v57',)","in ('onpolicy_refresh_v60',)")
        s=s.replace('Best fully evaluated v56 checkpoint; fresh AdamW at unchanged representation/rule rates','Best v57 step1000 checkpoint; fresh AdamW at unchanged rates; latest-policy successor data added')
        s=s.replace('Fully evaluated v56 source; add task-supervised equivalence of successful terminal images. Runtime remains unchanged exported hard-state rules.','v57 best1000; update coverage using measured successors from current symbolic policy. All architecture, loss weights and runtime behavior unchanged.')
        s=s.replace('if rank==0:atomic(out/\'manifest.json\',manifest)',"manifest.update(successor_refresh_round=3,source_navigation_successes=26,source_navigation_episodes=80,\n        research_reference='https://icml.cc/2012/papers/833.pdf',\n        intervention='Same v57 architecture/objectives/learning rates; half branch samples use new v57-policy outcomes, half use cumulative older outcomes',\n        source_selection='Best full-dev80; fixed finalfixed80 never used')\n    if rank==0:atomic(out/'manifest.json',manifest)")
    if name=='supervisor.py':
        s=s.replace("'--updates','2000'","'--updates','1000'").replace('29757','29760')
    (HERE/name).write_text(s,encoding='utf-8')
s=(BASE/'goal_equivalence_v57/data.py').read_text(encoding='utf-8')
s=s.replace("new.name=='onpolicy_successors_v54'","new.name=='onpolicy_successors_v60'")
s=s.replace("for extra in (ROOT/'onpolicy_successors_v40',new):","for extra in (ROOT/'onpolicy_successors_v40',ROOT/'onpolicy_successors_v54',new):")
(HERE/'data.py').write_text(s,encoding='utf-8')
s=(BASE/'goal_equivalence_v57/check_contract.py').read_text(encoding='utf-8')
s=s.replace("ROOT/'diagnostic_temporal_symbols_v55/source.pt'","ROOT/'onpolicy_successors_v60/source.pt'")
s=s.replace("pilot=torch.load(ROOT/'diagnostic_goal_equivalence_v57/probe.pt',map_location='cpu',weights_only=False)\nassert pilot['source']['sha256']==commit['sha256']\n",'')
s=s.replace("model.load_state_dict(pilot['model'],strict=True)","model.load_state_dict(saved['model'],strict=True)")
s=s.replace('onpolicy_successors_v54','onpolicy_successors_v60').replace('diagnostic_goal_equivalence_v57','diagnostic_onpolicy_refresh_v60')
(HERE/'check_contract.py').write_text(s,encoding='utf-8')
print('Prepared1000-update refinement; not launched. Final source and data audit required before launch.')
