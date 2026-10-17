import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('snapshot');a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'predictive_rules';snapshot=Path(a.snapshot);check=json.loads(snapshot.read_text(encoding='utf-8-sig'))
progress=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
progress.update(last_verified_update=check['status']['update'],last_verified_pid=check['status']['pid'],
    last_observed_state=check['status']['state'],last_remote_check_unix=check['time'],
    last_local_check_unix=snapshot.stat().st_mtime,next_regular_local_check_not_before_unix=snapshot.stat().st_mtime+660,
    last_record_update_unix=time.time())
progress['active_runs']['visual_graph_subgoal_v5'].update(latest_status=check['status'],latest_checkpoint=check['checkpoint'],
    process_identity_verified=check.get('process_identity_verified',False),committed_hash_verified=check.get('committed_hash_verified',False),
    recent=check['recent'],validation=check['validation'])
if check['validation']:
    best=max(check['validation'],key=lambda r:r['successes']);latest=check['validation'][-1]
    keys=['update','successes','episodes','mean_steps','planning_failures','exact_plan_found_rate','boundary_gate_violations']
    progress['current_best_development']={'run':'visual_graph_subgoal_v5',**{k:best[k] for k in keys if k in best}}
    progress['latest_completed_development']={'run':'visual_graph_subgoal_v5',**{k:latest[k] for k in keys if k in latest}}
progress['current_turn_classification']='progress: isolated inference cost and obtained an exact-token CUDA-graph speedup; evaluated true navigation; diagnosed inadequate predictive state learning'
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'update':check['status']['update'],'next_check_local':progress['next_regular_local_check_not_before_unix']}))
