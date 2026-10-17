"""Read completed saved traces only; no simulator/model execution."""
import json, statistics, collections, tarfile
from pathlib import Path

root=Path('/home/user/xjy/nesy_vla_20261010')
run=root/'runs/full_v1'
source=Path('/home/user/xjy/lite3_photo_world_20260927/lab_rl64/data_photo25')
read=lambda p:json.loads(p.read_text())
bank={r['episode_id']:r for r in read(source/'fixed80.json')}
completed=sorted(p for p in (run/'evaluations').glob('step_*') if (p/'fixed80/summary.json').exists())
step=completed[-1]
out=root/'runs/success_analysis';out.mkdir(exist_ok=True)

def row(path):
    d=read(path);steps=d['steps'];meta=bank.get(d['episode_id'],{})
    r={k:v for k,v in d.items() if k!='steps'}
    r.update(path=str(path.parent),actions=[s['action'] for s in steps],
        collisions=[i+1 for i,s in enumerate(steps) if s['collision']],
        visual_changes=[s['feedback'][1] for s in steps],
        versions=[s['version'] for s in steps],
        prediction_errors=[s['prediction_error'] for s in steps],
        action_probability=[s['support'][s['action']]/sum(s['support']) for s in steps],
        stop_probability=[s['support'][0]/sum(s['support']) for s in steps],
        observed_truth=[s['observed_truth'] for s in steps],
        belief=[s['belief'] for s in steps],
        initial_metadata=meta)
    return r

fixed=[row(p) for p in sorted((step/'fixed80').glob('*/trajectory.json'))]
training=[]
trace_updates=sorted((run/'traces').glob('update_*'))[-10:]
for folder in trace_updates:
    for p in sorted(folder.glob('*/trajectory.json')):
        d=row(p);d['update']=int(folder.name.split('_')[-1]);training.append(d)
history={p.name:read(p/'fixed80/summary.json') for p in completed}
result={'evaluated_checkpoint':step.name,'status_at_analysis':read(run/'status.json'),
        'history':history,'fixed':fixed,'training_saved_traces_last10_updates':training}
(out/'analysis.json').write_text(json.dumps(result))
successes=[r for r in fixed if r['metrics'].get('success')]
selected=successes[:]
selected += [r for r in fixed if not r['metrics'].get('success')][:3]
with tarfile.open(out/'cases.tar.gz','w:gz') as tar:
    for r in selected:
        p=Path(r['path']);n=len(r['actions'])
        for i in sorted(set([0,max(0,n//2),max(0,n-1)])):
            image=p/f'frame_{i:03d}.jpg'
            tar.add(image,arcname=r['episode_id']+'/'+image.name)
        for name in ['trajectory.json','program_000.json','program_001.json','program_002.json','program_003.json']:
            if (p/name).exists():tar.add(p/name,arcname=r['episode_id']+'/'+name)
print(json.dumps({'step':step.name,'summary':history[step.name],
    'successes':[{'episode':r['episode_id'],'group':r['distance_group'],'domain':r['domain'],
                  'actions':r['actions'],'collisions':len(r['collisions']),'metrics':r['metrics']}
                 for r in successes],
    'example_metadata':fixed[0]['initial_metadata'],
    'saved_training_traces':len(training),'saved_training_successes':sum(bool(r['metrics'].get('success')) for r in training),
    'analysis_bytes':(out/'analysis.json').stat().st_size,'cases_bytes':(out/'cases.tar.gz').stat().st_size},indent=2))
