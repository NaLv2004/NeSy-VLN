import json,time,subprocess
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
answer=dict(time=time.time(),runs={})
for name in ('expanded_dynamics_v71','expanded_history_v76','memory_dynamics_v77','joint_memory_dynamics_v78'):
    out=ROOT/('run_'+name);row={}
    for fn in ('supervisor.json','status_rank0.json','status_rank1.json','result.json','capacity.json','evaluation_repair.json'):
        p=out/fn
        if p.exists():
            record=json.loads(p.read_text())
            if record.get('pid'):
                proc=Path(f"/proc/{record['pid']}/cmdline")
                record['actual_cmdline']=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
            row[fn]=record
    for fn in ('training.log','evaluation_repair.log','rank0.jsonl','heldout.jsonl'):
        p=out/fn
        if p.exists():row[fn]=p.read_text(errors='replace').splitlines()[-2:]
    answer['runs'][name]=row
answer['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
out=ROOT/'run_expanded_dynamics_v71/development_best';traces=list(out.glob('*/trace.json'))
audit=dict(episodes=len(traces),successes=0,actions=0,violations=[])
for p in traces:
    row=json.loads(p.read_text());steps=row['steps'];audit['successes']+=bool(row['metrics'].get('success'));audit['actions']+=len(steps)
    if len(steps)>32:audit['violations'].append([p.parent.name,'too many actions'])
    for j,s in enumerate(steps):
        proof=s['proof']
        if s['action']!=proof['actions'][0] or proof['states'][-1]!=proof['goal'] or proof['actions'][-1]!=0:
            audit['violations'].append([p.parent.name,j,'action/proof mismatch'])
        if j and not s['renewed'] and s['memory']!=steps[j-1]['predicted_memory']:
            audit['violations'].append([p.parent.name,j,'memory overwritten'])
    if row['metrics'].get('success') and (not steps or steps[-1]['action']!=0):audit['violations'].append([p.parent.name,'success without stop'])
answer['v71_trace_audit']=audit
(ROOT/'memory_dynamics_observation.json').write_text(json.dumps(answer,indent=2));print(json.dumps(answer))
