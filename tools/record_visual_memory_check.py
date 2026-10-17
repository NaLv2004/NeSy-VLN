import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('snapshot');a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'predictive_rules';path=Path(a.snapshot)
check=json.loads(path.read_text(encoding='utf-8-sig'));progress=json.loads((root/'goal_progress.json').read_text())
progress.update(last_verified_update=check['status']['update'],last_verified_pid=check['status']['pid'],
    last_observed_state=check['status']['state'],last_remote_check_unix=check['time'],last_local_check_unix=path.stat().st_mtime,
    next_regular_local_check_not_before_unix=path.stat().st_mtime+660,next_regular_check_not_before_unix=check['time']+660,
    last_record_update_unix=time.time())
progress['active_runs']['visual_memory_subgoal_v6'].update(latest_status=check['status'],latest_checkpoint=check['checkpoint'],
    process_identity_verified=check.get('process_identity_verified',False),committed_hash_verified=check.get('committed_hash_verified',False),
    recent=check['recent'],validation=check['validation'])
if check['validation']:
    keys=['update','successes','episodes','mean_steps','planning_failures','exact_plan_found_rate','boundary_gate_violations']
    for name,row in [('current_best_development',max(check['validation'],key=lambda r:r['successes'])),('latest_completed_development',check['validation'][-1])]:
        progress[name]=dict(run='visual_memory_subgoal_v6',**{k:row[k] for k in keys if k in row})
progress['current_turn_classification']='progress: verified v6 updates and development outcomes; audited rule identity errors; implemented and tested definition-conditioned, state-renaming-equivariant complete rule generation'
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(update=check['status']['update'],next_check=progress['next_regular_local_check_not_before_unix'])))
