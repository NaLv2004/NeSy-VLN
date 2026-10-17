"""Bounded read-only status for the corrected experiment and its data jobs."""
import json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
report=dict(time=time.time(),runs={})
for name in ['run_anonymous_v5','run_v1','run_memory_v2','run_outcomes_v3',
             'counterfactual_own_policy_v1/shard_0','counterfactual_own_policy_v1/shard_1','pilot_planning_alignment',
             'run_anonymous_v6','counterfactual_anonymous_v5/shard_0','counterfactual_anonymous_v5/shard_1']:
    path=root/name;entry={}
    for filename in ['launcher.json','status.json','latest.COMMITTED.json','best_development.json','summary.json']:
        if (path/filename).exists():entry[filename]=json.loads((path/filename).read_text())
    if 'launcher.json' in entry:
        pid=entry['launcher.json']['pid'];entry['process']=subprocess.run(['ps','-p',str(pid),'-o','pid,stat,etime,pcpu,args'],capture_output=True,text=True).stdout
    if (path/'driver.log').exists():entry['log_tail']=(path/'driver.log').read_text(errors='replace')[-2000:]
    if (path/'validation.jsonl').exists():
        rows=[json.loads(line) for line in (path/'validation.jsonl').read_text().splitlines() if line]
        entry['validations']=[{key:r[key] for key in ['update','successes','episodes','collision_action_rate','neural_calls_outside_boundaries']} for r in rows]
    report['runs'][name]=entry
report['own_policy_files']=len(list((root/'counterfactual_own_policy_v1').glob('*.npz')))
report['anonymous_policy_files']=len(list((root/'counterfactual_anonymous_v5').glob('*.npz')))
report['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
(root/'anonymous_snapshot.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(time=report['time'],gpus=report['gpus'],anonymous_policy_files=report['anonymous_policy_files'],
    runs={name:dict(status=value.get('status.json'),process=value.get('process'),validations=value.get('validations'))
          for name,value in report['runs'].items() if name.startswith(('run_anonymous_','counterfactual_anonymous_'))}),indent=2))
