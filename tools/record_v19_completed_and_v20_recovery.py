import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules';e=root/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
p=read(root/'goal_progress.json');final=read(e/'raw_boundary_v19_final_check.json')
assert not final['process_alive'] and final['status']['state']=='training_budget_complete' and final['committed_hash_verified']
record=p['active_runs']['raw_boundary_v19'];record.update(last_observed_state='training_budget_complete',last_observed_update=600,
    process_alive=False,checkpoint=final['checkpoint'],development_results=final['validation'],recent=final['recent'],
    decision='Not promoted: best3/80 development versus5/80 prior exact-goal candidate and43/80 reference.')
p['raw_boundary_coverage_v19']=read(e/'raw_boundary_coverage_v19.json')
p['raw_boundary_v19_navigation_audit']=read(e/'raw_boundary_v19_navigation_audit.log')
startup=read(e/'replayed_boundary_v20_startup.json')
p['active_runs']['replayed_boundary_features_v20']=dict(path='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/replayed_boundary_features_v20',
    source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_replay_boundary_v20',scope='Training data recovery, not navigation training',
    launcher=read(e/'replayed_boundary_v20_launcher.json'),last_verified=startup,preflight=read(e/'replayed_boundary_preflight_v20.json'))
p.update(last_verified_update=600,last_verified_pid=None,last_remote_check_unix=startup['time'],
    next_regular_check_not_before_unix=startup['time']+660,active_scope='Recover exact raw boundary inputs for existing training-only transition banks',
    goal_status='active',completion_proven=False)
p['pending']=[
    'Wait on the existing four v20 replay process handles; do not duplicate/restart. Completion is event-driven, with no repeated status polling.',
    'Commit the cache only after every shard reports complete and all8066 valid transition states have exact canonical causal-history matches.',
    'Preflight the replay-input loader; then run the otherwise matched600-update raw-boundary candidate on GPUs3/4/5 with unchanged development80 onGPU7 every200.',
    'Do not claim data coverage, loss, proof reachability or component accuracy as navigation success. V19 completed2/3/3 successes of80; best prior exact-goal is5/80, reference43/80.',
    'Keep immutable checkpoints, original data, fixed partitions and unrelated GPU0/1/2/6 jobs. No maps, semantic labels, reward/value tables, action heads or neural corrections within a block.',
    'Final fixed80 remains untouched; complete navigation and speed parity are unachieved. Goal remains active.'
]
p['previous_turn_classification']='progress: completed v19 navigation result, diagnosed missing raw inputs and verified exact-action image recovery'
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Recorded completed navigation and active data recovery; goal not complete')
