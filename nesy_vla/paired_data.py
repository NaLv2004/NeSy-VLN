"""Validate extra successful paired-goal demonstrations without changing splits."""
import hashlib,json
from pathlib import Path


def load_paired_demonstrations(root,base_episodes,allowed,source_hashes):
    root=Path(root).resolve()
    read=lambda name:json.loads((root/name).read_text())
    complete=read('COMPLETE');provenance=read('source.json')
    if complete['provenance']!=provenance:raise ValueError('Paired provenance changed')
    for name in ['train.json','fixed80.json']:
        if provenance['source_hashes'][name]!=source_hashes[name]:raise ValueError('Paired source split changed')
    if set(provenance['train_layouts'])!=set(allowed):raise ValueError('Paired training layout whitelist differs')
    for name in ['examples','episodes','pairs']:
        digest=hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest()
        if digest!=complete[name+'_sha256']:raise ValueError('Paired manifest checksum differs: '+name)
    pairs=read('pairs.json');episodes=read('episodes.json');rows=read('examples.json');extra={};expected=[]
    for pair in pairs:
        origin=base_episodes.get(pair['origin'])
        if origin is None or origin['scene'] not in allowed:raise ValueError('Paired origin outside training split')
        if not pair['both_successful'] or not pair['same_initial_rgb']:raise ValueError('Unverified paired demonstration')
        if len(pair['branches'])!=2:raise ValueError('Paired record must have two branches')
        first_hashes=[];categories=[]
        for branch in pair['branches']:
            ep=branch['episode'];frames=branch['frames'];categories.append(ep['category'])
            if not branch['success'] or not frames or frames[-1]['action']!=0:raise ValueError('Unsuccessful paired branch')
            for key in ['scene','start_position','start_rotation']:
                if ep[key]!=origin[key]:raise ValueError('Paired start differs from its original training episode')
            if ep['episode_id'] in base_episodes or ep['episode_id'] in extra:raise ValueError('Paired episode ID collision')
            extra[ep['episode_id']]=ep
            for frame in frames:
                if frame['episode_id']!=ep['episode_id'] or frame['category']!=ep['category']:raise ValueError('Frame goal mismatch')
                path=(root/frame['image']).resolve()
                if not path.is_relative_to(root):raise ValueError('Paired image path escaped its root')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=frame['sha256']:raise ValueError('Paired image checksum differs')
            first_hashes.append(frames[0]['sha256']);expected.extend(frames)
        if len(set(categories))!=2 or len(set(first_hashes))!=1:raise ValueError('Paired goals or image equality invalid')
    if episodes!=list(extra.values()) or rows!=expected:raise ValueError('Paired flat dataset does not match validated records')
    rows=[dict(r,_image_root=str(root),paired_goal_demonstration=True) for r in rows]
    return extra,rows,{'root':str(root),'pairs':len(pairs),'episodes':len(extra),'frames':len(rows),
        'examples_sha256':complete['examples_sha256'],'source_hashes':provenance['source_hashes']}
