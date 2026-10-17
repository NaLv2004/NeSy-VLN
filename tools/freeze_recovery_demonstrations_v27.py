"""Freeze observed prefixes and successful recoveries; never label a random action expert."""
import hashlib,json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');reference=Path('/home/user/xjy/strips_wm_navigation_20261009')
out=base/'data/recovery_demonstrations_v27';assert not out.exists()
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((reference/'research/split_lock.json').read_text());cfg=json.loads((reference/'configs/experiment.json').read_text())
source=Path(cfg['source_data']);allowed=set(lock['train_scenes'])-set(lock['development_scenes'])
for name in ['train.json','fixed80.json']:assert digest(source/name)==lock['hashes'][name]
episodes={r['episode_id']:r for r in json.loads((source/'train.json').read_text())}
original_ids={r['episode_id'] for r in json.loads((base/'data/lossless_demonstrations_v7/examples.json').read_text())}
records=[];rows=[]
for marker in sorted((reference/'data/native_recovery_v3').glob('rank_*/recovery/*/COMMITTED')):
    commit=json.loads(marker.read_text());audit_path=marker.parent/'audit.json';audit=json.loads(audit_path.read_text())
    if not commit['success']:continue
    eid=marker.parent.name;episode=episodes[eid]
    assert eid not in original_ids and episode['scene'] in allowed and audit['scene']==episode['scene']
    examples=json.loads((marker.parent/'examples.json').read_text());trace=audit['trace'];prefix=audit['random_prefix_actions']
    assert trace[-1]['success'] and trace[-1]['action']==0 and len(trace)<=32
    history=examples[-1]['history'];assert len(history)==len(trace)
    assert examples[-1]['previous_actions']==[r['action'] for r in trace[:-1]]
    by_image={r['image']:r for r in examples};frames=[]
    for i,(image,event) in enumerate(zip(history,trace)):
        path=(reference/'data'/image).resolve();assert path.is_relative_to(reference/'data')
        supervised=i>=len(prefix)
        assert bool(event['expert_labeled'])==supervised
        if supervised:
            row=by_image[image]
            assert row['action']==event['action'] and row['previous_actions']==[t['action'] for t in trace[:i]]
            assert row['history']==history[:i+1]
        else:assert event['action']==prefix[i] and image not in by_image
        frame={'episode_id':eid,'category':episode['category'],'action':event['action'],'image':image,
               'sha256':digest(path),'supervision_mask':supervised,'public_feedback':{'collision':bool(event['collision'])}}
        frames.append(frame);rows.append(frame)
    records.append({'episode':episode,'prefix_actions':prefix,'success':True,'frames':frames,
        'source_audit':str(audit_path),'source_audit_sha256':digest(audit_path)})
assert len(records)==400
provenance={'time':time.time(),'source_hashes':lock['hashes'],'train_layouts':sorted(allowed),
    'image_root':str(reference/'data'),'episode_count':len(records),'frames':len(rows),
    'context_only_frames':sum(not r['supervision_mask'] for r in rows),'supervised_frames':sum(r['supervision_mask'] for r in rows),
    'prefix_actions_are_not_supervision':True,'no_manual_predicate_labels':True,'fixed_data_used':False,
    'source_recovery_audit_sha256':digest(base/'development_recovery_audit_v26.json')}
out.mkdir(parents=True)
for name,value in [('source',provenance),('recoveries',records),('examples',rows)]:
    (out/(name+'.json')).write_text(json.dumps(value,indent=2))
complete={'provenance':provenance,'recoveries_sha256':digest(out/'recoveries.json'),'examples_sha256':digest(out/'examples.json')}
(out/'COMPLETE').write_text(json.dumps(complete,indent=2));print(json.dumps(complete,indent=2))
