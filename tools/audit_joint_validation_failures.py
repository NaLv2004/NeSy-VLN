import json,tarfile,collections,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010')
def summarize(rows):
    counts=collections.Counter();actions=collections.Counter();failed=[]
    for row in rows:
        steps=row['steps'];success=bool(row['metrics'].get('success'));stop=bool(steps and steps[-1]['action']==0)
        counts['episodes']+=1;counts['successes']+=success
        counts['failed_stop']+=not success and stop;counts['failed_horizon']+=not success and len(steps)>=32 and not stop
        counts['failed_stop_at_first_action']+=not success and stop and len(steps)==1
        actions.update(s['action'] for s in steps)
        if not success:failed.append({'id':row['episode_id'],'steps':len(steps),'stop':stop,'metrics':row['metrics']})
    return {'counts':dict(counts),'action_counts':dict(actions),'failures':failed}
old=base/'evidence_archives/completed_evaluations/symbolic_robust_v7_runs_robust_multieffect_step_000040_development.tar.gz'
with tarfile.open(old) as archive:source=[json.load(archive.extractfile(m)) for m in archive if m.name.endswith('/trajectory.json')]
folder=base/'symbolic_joint_v9/runs/joint/evaluations/step_000010/development'
current=[json.loads(p.read_text()) for p in folder.glob('*/trajectory.json')]
assert len(source)==len(current)==80 and {r['episode_id'] for r in source}=={r['episode_id'] for r in current}
result={'time':time.time(),'same_development_tasks':True,'source_v7_step40':summarize(source),'joint_v9_step10':summarize(current)}
out=base/'symbolic_joint_v9/development10_failure_audit.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps({name:{k:v for k,v in result[name].items() if k!='failures'} for name in ['source_v7_step40','joint_v9_step10']},indent=2))
print(json.dumps({'example_new_failure':result['joint_v9_step10']['failures'][0]}))
