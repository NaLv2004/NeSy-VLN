"""Read-only snapshot of the isolated subgoal experiments."""
import json,time,hashlib,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
def read(path):
    return json.loads(path.read_text()) if path.exists() else None
result={'time':time.time(),'runs':{}}
for name in ['run_subgoal_v1','run_graph_subgoal_v3','run_graph_subgoal_v4']:
    folder=root/name
    info={key:read(folder/file) for key,file in [('status','status.json'),('launcher','launcher.json'),('retired','retired.json'),('checkpoint','latest.COMMITTED.json'),('evaluation','evaluation_launcher.json')]}
    all_results=[json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()] if (folder/'validation.jsonl').exists() else []
    info['validation']=[{k:r[k] for k in ['update','successes','episodes','mean_steps','planning_failures','exact_plan_found_rate','collision_action_rate'] if k in r} for r in all_results]
    # A retired trainer may not have collected an already-running evaluation.
    info['uncollected_summaries']=[]
    collected={r['update'] for r in all_results}
    for path in sorted(folder.glob('development_step_*/summary.json')):
        summary=read(path)
        if summary.get('update') not in collected:
            info['uncollected_summaries'].append({k:summary[k] for k in ['update','successes','episodes','mean_steps','planning_failures'] if k in summary})
    if (folder/'train.jsonl').exists():
        lines=(folder/'train.jsonl').read_text().splitlines();rows=[json.loads(line) for line in lines[-100:]]
        info['recent']={'count':len(rows),'mean_seconds':sum(x['seconds'] for x in rows)/len(rows),'first_update':rows[0]['update'],'last_update':rows[-1]['update']}
    if name=='run_graph_subgoal_v4':
        info['driver_tail']=(folder/'driver.log').read_text(errors='replace')[-1400:] if (folder/'driver.log').exists() else ''
        if info['checkpoint']:
            checkpoint=folder/info['checkpoint']['path'];actual=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            after=read(folder/'latest.COMMITTED.json')
            info['committed_hash_verified']=actual==info['checkpoint']['sha256'] and after==info['checkpoint']
    result['runs'][name]=info
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
result['compute_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
print(json.dumps(result))
