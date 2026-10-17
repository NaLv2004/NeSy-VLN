"""Recreate existing successful training demonstrations with exact online pixels.

Resumable per episode, with source provenance, collision feedback and replay
success verified. This creates neither predicates nor navigation maps.
"""
import argparse,hashlib,io,json,os,sys,time
from pathlib import Path
package_root=Path(__file__).resolve().parent
if not (package_root/'nesy_vla').is_dir():package_root=package_root.parent
sys.path.insert(0,str(package_root))
import numpy as np
from PIL import Image
from nesy_vla.runtime import setup,DataBank,atomic,pil,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=3)
    p.add_argument('--gpu',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    root=Path('/home/user/xjy/nesy_vla_20261010')
    cfg=json.loads((root/'symbolic_loss_v6/configs/loss_reference.json').read_text());setup(cfg)
    bank=DataBank(cfg);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=json.loads((out/'source.json').read_text())
    assert source['demonstration_hash']==bank.provenance['demonstration_hash']
    assert source['fixed_hash']==bank.provenance['fixed_hash'] and source['train_hash']==bank.provenance['train_hash']
    selected=sorted(bank.demo_ids)[args.shard::args.shards]
    sim=bank.simulator(out/f'sim_{args.shard}.log',0);results=[];started=time.time()
    try:
        for episode_id in selected:
            folder=out/'episodes'/episode_id;marker=folder/'COMMITTED.json'
            if marker.exists():
                previous=json.loads(marker.read_text())
                assert previous['source_hash']==source['demonstration_hash']
                assert all((out/f['image']).is_file() and sha(out/f['image'])==f['sha256'] for f in previous['frames'])
                results.append(previous);continue
            folder.mkdir(parents=True,exist_ok=True)
            episode=bank.by_id[episode_id];rows=bank.demos[episode_id];frames=[];errors=[]
            assert episode['scene'] in bank.provenance['train_layouts']
            reply=sim.call('reset',episode=episode)
            for index,row in enumerate(rows):
                image=pil(reply['rgb']);encoded=io.BytesIO()
                image.resize((228,228),Image.Resampling.BILINEAR).save(encoded,format='JPEG',quality=92)
                encoded.seek(0);old=Image.open(bank.demo_image_path(row)).convert('RGB')
                regenerated=Image.open(encoded).convert('RGB')
                errors.append(float(np.abs(np.asarray(regenerated).astype(float)-np.asarray(old).astype(float)).mean()))
                filename=folder/f'frame_{index:03d}.png';temporary=filename.with_suffix('.tmp')
                image.save(temporary,format='PNG',compress_level=3);temporary.replace(filename)
                reply=sim.call('step',action=row['action'])
                public=reply.get('feedback',{})
                frames.append({'image':str(filename.relative_to(out)),'sha256':sha(filename),
                    'category':row['category'],'episode_id':episode_id,'action':row['action'],
                    'public_feedback':{'collision':bool(public['collision'])} if 'collision' in public else {}})
                if reply.get('done'):break
            successful=bool(reply.get('metrics',{}).get('success')) and len(frames)==len(rows) and max(errors)<2.
            record={'episode_id':episode_id,'source_hash':source['demonstration_hash'],'frames':frames,
                'success':successful,'metrics':reply.get('metrics',{}),'max_recorded_pixel_mae':max(errors),
                'expected_steps':len(rows),'scene':episode['scene'],'group':{k:episode[k] for k in ['domain','category','distance_group']}}
            atomic(marker,record);results.append(record)
            atomic(out/f'progress_{args.shard}.json',{'pid':os.getpid(),'time':time.time(),'shard':args.shard,
                'completed':len(results),'total':len(selected),'successes':sum(r['success'] for r in results),
                'seconds':time.time()-started})
    finally:sim.close()
    atomic(out/f'shard_{args.shard}.json',{'source_hash':source['demonstration_hash'],'episodes':results})
    print(json.dumps({'shard':args.shard,'episodes':len(results),'successes':sum(r['success'] for r in results)}))


if __name__=='__main__':main()
