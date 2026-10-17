import collections,json,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_reward_v11');result={}
for name in ['original','bounded_repeat_stop40']:
    result[name]={}
    for suite in ['development','fixed80']:
        folder=root/'runs'/name/'evaluations/step_000010'/suite
        rows=[json.loads(p.read_text()) for p in folder.glob('*/trajectory.json')]
        assert len(rows)==80
        counts=collections.Counter();counts['episodes']=len(rows)
        for row in rows:
            steps=row['steps'];success=bool(row['metrics'].get('success'));stop=bool(steps and steps[-1]['action']==0)
            counts['success']+=success;counts['failed_stop']+=stop and not success
            counts['failed_horizon']+=len(steps)==32 and not stop and not success
            counts['failed_first_stop']+=len(steps)==1 and stop and not success
            counts['collision_steps']+=sum(bool(s.get('collision')) for s in steps)
            counts['actions']+=len(steps)
        result[name][suite]=dict(counts)
record={'time':time.time(),'matched_checkpoint_step':10,'results':result}
(root/'step10_failure_comparison.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
