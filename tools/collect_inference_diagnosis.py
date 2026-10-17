"""Read-only snapshot of the already completed step-30 evaluation."""
import json,tarfile,collections
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010');run=root/'runs/full_v1'
ref=Path('/home/user/xjy/strips_wm_navigation_20261009')
out=root/'runs/inference_diagnosis_step30';out.mkdir(exist_ok=True)
with tarfile.open(out/'evidence.tar.gz','w:gz') as tar:
    for p in sorted((run/'evaluations/step_000030/fixed80').glob('*/*.json')):
        tar.add(p,arcname='fixed80/'+p.parent.name+'/'+p.name)
    for p in run.glob('rank*.jsonl'):tar.add(p,arcname='logs/'+p.name)
    for name in ['config.json','status.json']:
        tar.add(run/name,arcname='logs/'+name)
    for name in ['generator.py','training.py','logic.py','policy.py']:
        tar.add(root/'nesy_vla'/name,arcname='code/'+name)
examples=json.loads((ref/'data/search_cost_supervision_v2/examples.json').read_text())
episodes=collections.defaultdict(list)
for r in examples:episodes[r['episode_id']].append(r)
for rows in episodes.values():rows.sort(key=lambda x:x['image'])
hist=collections.defaultdict(collections.Counter);weighted=collections.defaultdict(collections.Counter)
chunk_n=collections.Counter();lengths=collections.Counter()
for rows in episodes.values():
    rows=rows[:32];lengths[len(rows)]+=1
    for start in range(0,len(rows),10):
        chunk=rows[start:start+10];kind='initial' if start==0 else 'revision';chunk_n[kind]+=1
        for r in chunk:
            hist[kind][r['action']]+=1;weighted[kind][r['action']]+=1/len(chunk)
data={'episodes':len(episodes),'lengths':lengths,'raw_action_counts':hist,
      'chunk_equal_weight_action_counts':weighted,'chunks':chunk_n}
(out/'demonstration_distribution.json').write_text(json.dumps(data,indent=2))
print(json.dumps({'archive_bytes':(out/'evidence.tar.gz').stat().st_size,'demonstrations':data},indent=2))
