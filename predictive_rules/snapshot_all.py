"""Bounded read-only snapshot; scheduling remains >=600s for stable runs."""
import json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
report=dict(time=time.time(),runs={})
for name in ['run_v1','run_memory_v2','run_outcomes_v3','counterfactual_training_v1']:
    path=root/name;record={}
    for file in ['launcher.json','status.json','latest.COMMITTED.json','best_development.json','COMPLETE.json']:
        if (path/file).exists():record[file]=json.loads((path/file).read_text())
    launcher=record.get('launcher.json',{})
    if launcher:
        pid=launcher['pid'];record['process']=subprocess.run(['ps','-p',str(pid),'-o','pid,ppid,stat,etime,pcpu,pmem,args'],capture_output=True,text=True).stdout
    if (path/'validation.jsonl').exists():
        records=[json.loads(line) for line in (path/'validation.jsonl').read_text().splitlines() if line]
        record['validations']=[{key:r[key] for key in ['update','successes','episodes','collision_action_rate','seconds','neural_calls_outside_boundaries']} for r in records]
    if (path/'driver.log').exists():record['log_tail']=(path/'driver.log').read_text(errors='replace')[-1600:]
    report['runs'][name]=record
report['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
report['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
(root/'latest_snapshot.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
