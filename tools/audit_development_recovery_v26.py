import collections,hashlib,json,time,zlib
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');reference=Path('/home/user/xjy/strips_wm_navigation_20261009')
mean=lambda v:sum(v)/len(v) if v else None
def trajectories(folder):
    archive=folder/'ARCHIVED_TRACES.json'
    if archive.exists():
        index=json.loads(Path(json.loads(archive.read_text())['content_index']).read_text())
        for member in index['members']:
            if Path(member['path']).name!='trajectory.json':continue
            raw=zlib.decompress(Path(member['blob']).read_bytes())
            assert hashlib.sha256(raw).hexdigest()==member['sha256']
            yield json.loads(raw)
    else:
        for path in sorted(folder.glob('*/trajectory.json')):yield json.loads(path.read_text())
result={'time':time.time(),'fixed80_read':False,'datasets':{},'development':{}}
lock=json.loads((reference/'research/split_lock.json').read_text())
for name in ['branch_control','conjunctive']:
    folder=base/'symbolic_conjunctive_v23/runs'/name/'evaluations/step_000010/development'
    rows=list(trajectories(folder));assert len(rows)==80
    assert all(r['scene'] in lock['development_scenes'] for r in rows)
    groups={'after_collision':[],'after_free_motion':[]};per_episode=[]
    for trace in rows:
        repeats=0;collisions=0;revisions=[]
        for a,b in zip(trace['steps'],trace['steps'][1:]):
            if a['action']==0:continue
            different=[x!=y for x,y in zip(a['symbol_codes'],b['symbol_codes'])]
            row={'action_repeated':a['action']==b['action'],'symbols_identical':not any(different),
                'symbol_change_fraction':mean(different),'visual_change':a['feedback'][1],
                'revision_boundary':a['version']!=b['version']}
            groups['after_collision' if a['collision'] else 'after_free_motion'].append(row)
            if row['revision_boundary']:revisions.append(row)
            repeats+=row['action_repeated'];collisions+=a['collision']
        per_episode.append({'episode':trace['episode_id'],'success':trace['metrics'].get('success',False),
            'steps':len(trace['steps']),'actions':[r['action'] for r in trace['steps']],
            'collisions':sum(r['collision'] for r in trace['steps']),
            'revision_boundaries':revisions})
    result['development'][name]={'groups':{key:{'pairs':len(values),**{metric:mean([row[metric] for row in values])
        for metric in ['action_repeated','symbols_identical','symbol_change_fraction','visual_change','revision_boundary']}}
        for key,values in groups.items()},'episodes':per_episode}
cfg=json.loads((reference/'configs/experiment.json').read_text());lock=json.loads((reference/'research/split_lock.json').read_text())
source=Path(cfg['source_data']);allowed=set(lock['train_scenes'])-set(lock['development_scenes'])
for name in ['train.json','fixed80.json']:assert hashlib.sha256((source/name).read_bytes()).hexdigest()==lock['hashes'][name]
by_id={r['episode_id']:r for r in json.loads((source/'train.json').read_text())}
for name in ['search_cost_supervision_v2','search_cost_supervision_v3']:
    path=reference/'data'/name/'examples.json'
    if path.exists():
        data=json.loads(path.read_text());counts=collections.Counter(r['episode_id'] for r in data)
        assert all(by_id[eid]['scene'] in allowed for eid in counts)
        result['datasets'][name]={'episodes':len(counts),'frames':len(data),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'row_keys':list(data[0]),'has_public_collision_rows':sum('public_feedback' in r for r in data)}
counts=collections.Counter();examples=[];images=[]
for marker in sorted((reference/'data/native_recovery_v3').glob('rank_*/recovery/*/COMMITTED')):
    commit=json.loads(marker.read_text());audit=json.loads((marker.parent/'audit.json').read_text())
    assert audit['scene'] in allowed and audit['scene']==by_id[marker.parent.name]['scene']
    trace=audit['trace'];prefix=audit['random_prefix_actions'];counts['attempts']+=1;counts['successful']+=commit['success']
    counts['prefix_steps']+=len(prefix);counts['prefix_collisions']+=sum(r['collision'] for r in trace[:len(prefix)])
    counts['recovery_steps']+=max(0,len(trace)-len(prefix));counts['recovery_collisions']+=sum(r['collision'] for r in trace[len(prefix):])
    if commit['success']:
        rows=json.loads((marker.parent/'examples.json').read_text())
        assert rows[-1]['action']==0 and trace[-1]['success'] and len(rows[-1]['history'])==len(trace)<=32
        assert rows[-1]['previous_actions']==[r['action'] for r in trace[:-1]]
        counts['successful_with_prefix_collision']+=any(r['collision'] for r in trace[:len(prefix)])
        counts['successful_with_recovery_collision']+=any(r['collision'] for r in trace[len(prefix):])
        for file in rows[-1]['history']:assert (reference/'data'/file).is_file()
        if len(examples)<4:examples.append({'episode':marker.parent.name,'prefix':prefix,'steps':len(trace),'first_row':rows[0]})
result['recovery']={'counts':dict(counts),'examples':examples,'all_successful_histories_present':True,'all_training_scenes':True}
(base/'development_recovery_audit_v26.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'datasets':result['datasets'],'recovery':dict(counts),'development':{k:v['groups'] for k,v in result['development'].items()}},indent=2))
