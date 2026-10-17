"""Causal recovery demonstrations with random prefixes excluded from imitation."""
import hashlib,json
from pathlib import Path


def load_recovery_demonstrations(root,base_episodes,allowed,source_hashes,permitted_image_root):
    root=Path(root).resolve();read=lambda name:json.loads((root/name).read_text())
    complete=read('COMPLETE');source=read('source.json')
    if complete['provenance']!=source:raise ValueError('Recovery provenance changed')
    if source['source_hashes']!=source_hashes or set(source['train_layouts'])!=set(allowed):
        raise ValueError('Recovery source split differs')
    image_root=Path(source['image_root']).resolve()
    if image_root!=Path(permitted_image_root).resolve():raise ValueError('Unregistered recovery image root')
    for name in ['recoveries','examples']:
        if hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest()!=complete[name+'_sha256']:
            raise ValueError('Recovery manifest checksum differs')
    records=read('recoveries.json');expected=[];ids=[]
    for record in records:
        ep=record['episode'];eid=ep['episode_id'];origin=base_episodes.get(eid)
        if origin is None or origin!=ep or ep['scene'] not in allowed:raise ValueError('Recovery outside original training split')
        if eid in ids:raise ValueError('Duplicate recovery episode')
        frames=record['frames'];prefix=record['prefix_actions']
        if not record['success'] or not 0<len(prefix)<len(frames)<=32 or frames[-1]['action']!=0:
            raise ValueError('Recovery must finish successfully within32actions')
        for i,row in enumerate(frames):
            if row['episode_id']!=eid or row['category']!=ep['category']:raise ValueError('Recovery task mismatch')
            if type(row['supervision_mask']) is not bool or row['supervision_mask']!=(i>=len(prefix)):
                raise ValueError('Random prefix cannot be an expert label')
            if type(row['action']) is not int or not 0<=row['action']<17:raise ValueError('Invalid action token')
            if i<len(prefix) and row['action']!=prefix[i]:raise ValueError('Prefix action history changed')
            path=(image_root/row['image']).resolve()
            if not path.is_relative_to(image_root):raise ValueError('Recovery image escaped root')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Recovery image checksum differs')
        ids.append(eid);expected.extend(frames)
    rows=read('examples.json')
    if rows!=expected:raise ValueError('Recovery examples differ from causal trajectories')
    return ids,[dict(row,_image_root=str(image_root),recovery_demonstration=True) for row in rows],dict(
        root=str(root),image_root=str(image_root),episodes=len(ids),frames=len(rows),
        context_frames=sum(not r['supervision_mask'] for r in rows),source_hashes=source_hashes,
        examples_sha256=complete['examples_sha256'])
