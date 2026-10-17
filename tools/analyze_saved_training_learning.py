import json
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');monitor=json.loads((base/'research_monitor_20261011.json').read_text());result={}
for run in ['symbolic_task_v15/runs/paired_tasks','symbolic_branch_v17/runs/branch_current']:
    m=monitor['jobs'][run];end=m['last_update']['update'];rows=[json.loads(line) for line in (base/run/'rank0.jsonl').read_text().splitlines()]
    # Restrict to the already-observed monitor snapshot rather than polling live training.
    rows=[row for row in rows if row['update']<=end]
    result[run]={'config':json.loads((base/run/'config.json').read_text()),'updates':[],'evaluations':m['evaluations']}
    for row in rows:
        if row['update'] not in [1,5,10,20,30,40,end]:continue
        result[run]['updates'].append({k:row[k] for k in ['update','seconds','gradient_norm','gradient_by_module','losses']})
(base/'learning_snapshot_diagnosis.json').write_text(json.dumps(result,indent=2))
for value in result.values():
    value['config']={k:v for k,v in value['config'].items() if k in ['loss','evidence_only_grounder','spatial_encoding','robust_grounding','program','training_mode']}
    value['evaluations']={step:{suite:summary.get('successes') if summary else None for suite,summary in entries.items()} for step,entries in value['evaluations'].items()}
print(json.dumps(result,indent=2))
