import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules';e=root/'evidence'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
p=read(root/'goal_progress.json');final=read(e/'replayed_boundary_v20_completed.json')
assert not final['process_alive'] and final['committed_hash_verified'] and final['status']['state']=='training_budget_complete'
cache=read(e/'replayed_boundary_v20_manifest.json');assert cache['boundary_state_coverage']==1. and cache['canonical_patch_agreement']==1.
p['active_runs']['replayed_boundary_features_v20'].update(last_observed_state='complete',processes_alive=False,
    result={k:v for k,v in cache.items() if k not in {'records','frozen_vision'}},manifest=str(e/'replayed_boundary_v20_manifest.json'))
p['active_runs']['replayed_boundary_v20']=dict(path='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_replayed_boundary_v20',
    last_observed_state='training_budget_complete',last_observed_update=600,process_alive=False,
    checkpoint=final['checkpoint'],committed_hash_verified=True,development_results=final['validation'],recent=final['recent'],
    decision='Best4/80 does not establish improvement over prior5/80; no final fixed-suite evaluation')
p['raw_goal_derivations_v20']={k:v for k,v in read(e/'raw_goal_derivations_v20.json').items() if k!='rows'}
p['categorical_precision_v20']=read(e/'categorical_precision_v20.json')
p['rule_gradient_conflict_v21']=read(e/'rule_gradient_conflict_v21.json')
p['rule_gradient_conflict_v21']['interpretation']='Mixed mostly weak gradient alignment; not evidence of uniformly strong conflict. The next candidate tests cleaner training responsibilities, not a proven causal explanation.'
current=read(e/'separated_goal_v21_startup.json');assert current['process_alive'] and current['process_identity_verified'] and current['committed_hash_verified']
p['active_runs']['separated_goal_v21']=dict(path='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_separated_goal_v21',
    source=current['launcher']['source'],last_observed_state='training',last_observed_update=current['status']['update'],
    pid=current['status']['pid'],launcher_pid=current['launcher']['pid'],checkpoint=current['checkpoint'],committed_hash_verified=True,
    actual_optimizer_updates_confirmed=True,process_identity_verified=True,recent=current['recent'],
    preflight=read(e/'separated_goal_v21_preflight.json'),development_results=current['validation'],evaluation_error=current['evaluation_error'])
p.update(remote_run=p['active_runs']['separated_goal_v21']['path'],remote_source=current['launcher']['source'],
    last_verified_pid=current['status']['pid'],last_verified_update=current['status']['update'],
    last_remote_check_unix=current['time'],next_regular_check_not_before_unix=current['time']+660,
    active_scope='Full exact-goal navigation with separately trained dynamics and subgoal branches',goal_status='active',completion_proven=False)
p['pending']=[
    'Observe existing v21 process via a kernel process-exit wait or after the next660-second regular interval; never restart on transport timeout.',
    'V21 is1000updates, GPUs3/4/5,512demo windows+128transition states percard, dev80 at500/1000 onGPU7. It starts from immutablev20step600 with a fresh optimizer.',
    'Only actual one-step observed transitions train world rules. Separate goal context/head learn future discrete endpoint+proof action loss through detached rules/premises. Open-loop/inverse/reconstruction losses are monitor-only.',
    'Verify the full navigation results. Initial program equality and zero cross-branch gradients are verified, but navigation and end-to-end speed parity remain unproven.',
    'V20 recovery is complete:8066states,395234patch IDs,825bank/episode records,100%canonical agreement. Its navigation3/3/4of80 did not beat the prior5/80.',
    'The v20 training-only128window audit showed243/1033 one-step,189/1033 autonomous matches; precision and an oracle final-goal intervention did not explain/resolve the gap.',
    'Keep all checkpoints and data; train layouts100..115/118..121; development116/117/122/123; finalfixed200..207 untouched. Preserve other GPU0/1/2/6 processes.',
    'No manual semantic states, collision predicates, value/reward tables, maps, direct actors, approximate proofs, goal replacement or neural correction between0/10/20/30.',
    'Reference remains43/80 development,44/80fixed. Goal active until navigation and complete execution-speed criteria actually met.'
]
p['previous_turn_classification']='progress: canonical raw data recovery completed; full navigation measured; ruled out two simple explanations; gradient-isolated complete candidate verified training'
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(current_run=p['remote_run'],verified_update=p['last_verified_update'],next_regular_check=p['next_regular_check_not_before_unix'],goal_complete=False)))
