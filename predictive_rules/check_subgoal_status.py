"""A compact evidence snapshot; no restart or process mutation."""
import json,time,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
result=dict(time=time.time())
for name in ['run_subgoal_v1','subgoal_rule_inverse_trial']:
    out=root/name;row={}
    for file in ['launcher.json','status.json','latest.COMMITTED.json','candidate.COMMITTED.json','evaluation_error.json','training_diagnosis.json']:
        p=out/file
        if p.exists():
            value=json.loads(p.read_text())
            if file=='training_diagnosis.json':value={k:v for k,v in value.items() if k!='examples'}
            row[file]=value
    p=out/'validation.jsonl'
    if p.exists():
        keys=['update','successes','episodes','planning_failures','exact_plan_found_rate','mean_steps','seconds','boundary_gate_violations']
        row['validations']=[{k:v for k,v in json.loads(line).items() if k in keys} for line in p.read_text().splitlines()]
    p=out/'development/summary.json'
    if p.exists():row['development']=json.loads(p.read_text())
    if (out/'driver.log').exists():row['log_tail']=(out/'driver.log').read_text(errors='replace').splitlines()[-2:]
    if (out/'launcher.json').exists():
        pid=json.loads((out/'launcher.json').read_text())['pid']
        command=Path('/proc')/str(pid)/'cmdline'
        row['process']=dict(pid=pid,exists=command.exists(),command=command.read_bytes().replace(b'\0',b' ').decode() if command.exists() else None)
    result[name]=row
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
print(json.dumps(result,indent=2))
