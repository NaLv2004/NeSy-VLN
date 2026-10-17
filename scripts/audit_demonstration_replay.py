"""Verify training demonstrations by executing their exact stored motor actions.

This is a dataset audit, not a learned-policy evaluation. No maps or predicate
labels are generated, and no development/fixed tasks are selected.
"""
import io,json,os,random,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='6'
package_root=Path(__file__).resolve().parent
if not (package_root/'nesy_vla').is_dir():package_root=package_root.parent
sys.path.insert(0,str(package_root))
import numpy as np
from PIL import Image
from nesy_vla.runtime import setup,DataBank,atomic,pil

root=Path('/home/user/xjy/nesy_vla_20261010')
cfg=json.loads((root/'symbolic_loss_v6/configs/loss_reference.json').read_text());setup(cfg)
bank=DataBank(cfg);out=root/'diagnostics_20261011/demonstration_replay';out.mkdir(exist_ok=True)
assert not (out/'result.json').exists(),'Read the completed audit instead of repeating it'
groups={}
for episode_id in bank.demo_ids:
    ep=bank.by_id[episode_id]
    groups.setdefault((ep['domain'],ep['category'],ep['distance_group']),[]).append(episode_id)
rng=random.Random(20261011)
selected=[episode_id for key in sorted(groups) for episode_id in rng.sample(sorted(groups[key]),min(2,len(groups[key])))]
assert all(bank.by_id[i]['scene'] in bank.provenance['train_layouts'] for i in selected)
sim=bank.simulator(out/'sim.log',0);results=[]
try:
    for episode_id in selected:
        ep=bank.by_id[episode_id];rows=bank.demos[episode_id]
        reply=sim.call('reset',episode=ep);errors=[];actions=[]
        for row in rows:
            actual=pil(reply['rgb']).resize((228,228),Image.Resampling.BILINEAR)
            buffer=io.BytesIO();actual.save(buffer,format='JPEG',quality=92);buffer.seek(0)
            recreated=Image.open(buffer).convert('RGB')
            recorded=Image.open(bank.root/'data'/row['image']).convert('RGB')
            error=float(np.abs(np.asarray(recreated).astype(float)-np.asarray(recorded).astype(float)).mean())
            errors.append(error);actions.append(row['action'])
            reply=sim.call('step',action=row['action'])
            if reply.get('done'):break
        result={k:ep[k] for k in ['episode_id','scene','category','domain','distance_group']}
        result.update(success=bool(reply.get('metrics',{}).get('success')),
            expected_actions=len(rows),executed_actions=actions,pixel_mae=errors,
            first_frame_pixel_mae=errors[0],mean_frame_pixel_mae=sum(errors)/len(errors),
            final_metrics=reply.get('metrics',{}))
        results.append(result)
        atomic(out/'progress.json',{'pid':os.getpid(),'time':time.time(),'completed':len(results),'total':len(selected),
            'successes':sum(r['success'] for r in results)})
finally:sim.close()
summary={'episodes':len(results),'successes':sum(r['success'] for r in results),
    'mean_first_frame_pixel_mae':float(np.mean([r['first_frame_pixel_mae'] for r in results])),
    'mean_all_frame_pixel_mae':float(np.mean([v for r in results for v in r['pixel_mae']])),
    'failed_episode_ids':[r['episode_id'] for r in results if not r['success']],
    'qualification':'Replay of existing training labels; this success rate is not model performance.'}
atomic(out/'result.json',{'summary':summary,'episodes':results,'data_provenance':bank.provenance})
print(json.dumps(summary,indent=2))
