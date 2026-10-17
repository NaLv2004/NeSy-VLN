"""Matched RGB starts with independently successful demonstrations for each goal.

Uses only locked training layouts. Simulator geometry is private to collection;
learners receive RGB, goal text and executed opaque actions, never predicates.
"""
import argparse,copy,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-root',required=True)
    p.add_argument('--out',required=True);p.add_argument('--target-pairs',type=int,default=64)
    p.add_argument('--gpu',default='7');args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    reference=Path(args.reference_root);sys.path.insert(0,str(reference))
    from stripsnav.io import read_json,atomic_json,sha256
    from stripsnav.sim_adapter import NavigationEnvironment
    from stripsnav.actions import N_ACTIONS
    cfg=read_json(reference/'configs/experiment.json');cfg['training_horizon']=32
    lock=read_json(reference/'research/split_lock.json');source=Path(cfg['source_data'])
    for name in ['train.json','fixed80.json']:assert sha256(source/name)==lock['hashes'][name]
    allowed=set(lock['train_scenes'])-set(lock['development_scenes'])
    assert allowed.isdisjoint(lock['fixed_scenes'])
    train=[e for e in read_json(source/'train.json') if e['scene'] in allowed]
    categories=sorted(set(e['category'] for e in train));assert len(categories)==2
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    provenance={'source_hashes':lock['hashes'],'train_layouts':sorted(allowed),
        'development_layouts':lock['development_scenes'],'fixed_layouts':lock['fixed_scenes'],
        'horizon':32,'no_handwritten_predicates':True,'policy_input_fields':['image','category','action','public_feedback'],
        'teacher':'Existing training-only action probing oracle; no new map constructed',
        'selection':'Both trajectories must finish successfully. No learned-policy or held-out scores used.'}
    if (out/'source.json').exists():assert read_json(out/'source.json')==provenance
    else:atomic_json(out/'source.json',provenance)
    rng=random.Random(20261011);candidates=[]
    for scene in sorted(allowed):
        for distance in ['near','medium','far']:
            for category in categories:
                pool=[e for e in train if e['scene']==scene and e['category']==category and e['distance_group']==distance]
                candidates.append(rng.choice(pool))
    # Interleave layouts so an interrupted collection is not concentrated in one room.
    candidates=sorted(candidates,key=lambda e:(['near','medium','far'].index(e['distance_group']),e['category'],e['scene']))
    templates={(scene,category):next(e for e in train if e['scene']==scene and e['category']==category)
               for scene in allowed for category in categories}
    records=[];env=NavigationEnvironment(cfg,gpu=0);started=time.time()
    try:
        for origin in candidates:
            pair_id=origin['episode_id'];folder=out/'pairs'/pair_id;marker=folder/'COMMITTED.json'
            if marker.exists():records.append(read_json(marker))
            else:
                folder.mkdir(parents=True,exist_ok=True);branches=[]
                for category in categories:
                    ep=copy.deepcopy(origin);template=templates[(origin['scene'],category)]
                    for key in ['category','viewpoints','goal_object_names','description']:ep[key]=copy.deepcopy(template[key])
                    ep['episode_id']='paired_'+pair_id+'_'+category
                    rgb=env.reset(ep);first_digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
                    frames=[];initial=env.private_metrics();metric={}
                    for step in range(32):
                        action=0 if env.private_metrics()['eligible'] else env.oracle_action({a:env.probe(a) for a in range(1,N_ACTIONS)})
                        image=folder/category/f'frame_{step:03d}.png';image.parent.mkdir(exist_ok=True)
                        temp=image.with_suffix('.tmp');Image.fromarray(rgb).save(temp,format='PNG',compress_level=3);temp.replace(image)
                        response=env.step(action);rgb=response['rgb'];metric=response['metrics']
                        frames.append({'episode_id':ep['episode_id'],'category':category,'action':int(action),
                            'image':str(image.relative_to(out)),'sha256':sha256(image),
                            'public_feedback':{'collision':bool(response['feedback']['collision'])}})
                        if response['done']:break
                    branches.append({'episode':ep,'initial_rgb_sha256':first_digest,'initial_metrics':initial,
                        'success':bool(metric.get('success')),'metrics':metric,'frames':frames})
                same=branches[0]['initial_rgb_sha256']==branches[1]['initial_rgb_sha256']
                record={'pair_id':pair_id,'origin':origin['episode_id'],'scene':origin['scene'],
                    'same_initial_rgb':same,'both_successful':all(b['success'] for b in branches),
                    'first_actions_differ':branches[0]['frames'][0]['action']!=branches[1]['frames'][0]['action'],
                    'branches':branches,'time':time.time()}
                if not same:raise RuntimeError('Changing only the goal changed RGB; pair is invalid')
                atomic_json(marker,record);records.append(record)
            valid=[r for r in records if r['both_successful'] and r['same_initial_rgb']]
            atomic_json(out/'progress.json',{'state':'collecting','pid':os.getpid(),'time':time.time(),
                'attempted_pairs':len(records),'successful_pairs':len(valid),
                'different_first_action_pairs':sum(r['first_actions_differ'] for r in valid),
                'seconds':time.time()-started,'target_pairs':args.target_pairs})
            print(json.dumps({'pairs':len(records),'successful':len(valid),'different_first_actions':sum(r['first_actions_differ'] for r in valid)}),flush=True)
            if len(valid)>=args.target_pairs:break
    finally:env.close()
    valid=[r for r in records if r['both_successful'] and r['same_initial_rgb']]
    atomic_json(out/'pairs.json',valid)
    atomic_json(out/'examples.json',[f for r in valid for b in r['branches'] for f in b['frames']])
    atomic_json(out/'episodes.json',[b['episode'] for r in valid for b in r['branches']])
    atomic_json(out/'COMPLETE',{'attempted':len(records),'pairs':len(valid),
        'different_first_actions':sum(r['first_actions_differ'] for r in valid),
        'examples_sha256':sha256(out/'examples.json'),'pairs_sha256':sha256(out/'pairs.json'),
        'episodes_sha256':sha256(out/'episodes.json'),'provenance':provenance})


if __name__=='__main__':main()
