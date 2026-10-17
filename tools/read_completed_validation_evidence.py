import json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');result={'time':time.time(),'runs':{}}
for relative in ['symbolic_robust_v7/runs/robust_multieffect','symbolic_joint_v9/runs/joint',
    'symbolic_vision_v8/runs/frozen_convnext','symbolic_vision_v8/runs/frozen_qwen',
    'calibrated_vision_v10/runs/uncalibrated_control','calibrated_vision_v10/runs/calibrated_qwen']:
    rows=[]
    for step in sorted((base/relative/'evaluations').glob('step_*')):
        row={'update':int(step.name.split('_')[-1])}
        for suite in ['development','fixed80']:
            path=step/suite/'summary.json'
            if path.exists():
                summary=json.loads(path.read_text());row[suite]={k:summary[k] for k in ['successes','episodes','collision_action_rate','mean_steps']}
        if 'development' in row and 'fixed80' in row:rows.append(row)
    result['runs'][relative]=rows
best=max((dict(run=name,**row) for name,rows in result['runs'].items() for row in rows),key=lambda r:r['fixed80']['successes'])
result['best_observed_fixed80']=best
(base/'validation_evidence_20261011.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'best_observed_fixed80':best,'latest':{name:rows[-1] if rows else None for name,rows in result['runs'].items()}},indent=2))
