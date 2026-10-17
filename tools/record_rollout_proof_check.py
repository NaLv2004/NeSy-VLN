"""Record a verified run inspection without resetting experiment history."""
import argparse,json,time
from pathlib import Path,PurePosixPath
p=argparse.ArgumentParser();p.add_argument('snapshot');a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'predictive_rules';path=Path(a.snapshot)
check=json.loads(path.read_text(encoding='utf-8-sig'));progress=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
assert progress['architecture'] in {'anonymous_rollout_proof_subgoal_v8','anonymous_definition_conditioned_world_v12'}
run=PurePosixPath(progress['remote_run']).name.removeprefix('run_')
previous=progress.get('active_runs',{}).get(run,{}).get('latest_status',{}).get('update')
assert check['committed_hash_verified']
if check['process_alive']:assert check['process_identity_verified']
status=check['status']
progress.update(last_verified_update=status['update'],last_verified_pid=status['pid'],
    last_observed_state=status['state'],last_remote_check_unix=check['time'],last_local_check_unix=path.stat().st_mtime,
    next_regular_local_check_not_before_unix=path.stat().st_mtime+660,next_regular_check_not_before_unix=check['time']+660,
    last_record_update_unix=time.time())
progress['active_runs'][run].update(latest_status=status,latest_checkpoint=check['checkpoint'],
    process_identity_verified=check.get('process_identity_verified',False),committed_hash_verified=True,
    process_alive=check['process_alive'],recent=check['recent'],validation=check['validation'])
if check['validation']:
    keys=['update','successes','episodes','mean_steps','planning_failures','exact_plan_found_rate','boundary_gate_violations']
    for name,row in [('current_best_development',max(check['validation'],key=lambda r:r['successes'])),('latest_completed_development',check['validation'][-1])]:
        progress[name]=dict(run=run,**{k:row[k] for k in keys if k in row})
progress['current_turn_classification']=('progress: inspected actual optimizer updates, immutable checkpoint hash and navigation outcomes for '+run
    if previous!=status['update'] else 'verified wait: existing launcher identity is live and unchanged' if check['process_alive'] else
    'terminal state verified: no automatic restart or completion claim')
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(update=status['update'],state=status['state'],best=progress['current_best_development'],latest=progress['latest_completed_development'],next_check=progress['next_regular_local_check_not_before_unix'])))
