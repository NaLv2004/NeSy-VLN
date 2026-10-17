"""Frozen-policy development evaluation with the demonstration image transport.

The only treatment is RGB -> 228px -> JPEG92 -> ordinary model preprocessing.
Motor actions remain generated-rule inference over discrete learned predicates.
"""
import argparse,io,json,os,sys,time,hashlib
from pathlib import Path
package_root=Path(__file__).resolve().parent
if not (package_root/'nesy_vla').is_dir():package_root=package_root.parent
sys.path.insert(0,str(package_root))
import torch
from PIL import Image
from nesy_vla.runtime import setup,load_checkpoint,DataBank,atomic
from nesy_vla.model import NavigationModel
from nesy_vla import policy
from nesy_vla.reporting import save_trace
from nesy_vla.evaluation import aggregate


def demonstration_transport(image):
    output=io.BytesIO()
    image.resize((228,228),Image.Resampling.BILINEAR).save(output,format='JPEG',quality=92)
    output.seek(0)
    return Image.open(output).convert('RGB')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--step',type=int,default=30)
    parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(2)
    root=Path(args.run);cfg=json.loads((root/'config.json').read_text());setup(cfg)
    assert cfg['discrete_reasoning'] and cfg['generator_backend']=='symbolic_gru' and cfg['horizon']==32
    checkpoint=root/f'evaluations/step_{args.step:06d}/checkpoint';marker=json.loads((checkpoint/'COMMITTED').read_text())
    cfg['lossless_trace_images']=True
    model=NavigationModel(cfg).to('cuda').eval();load_checkpoint(checkpoint,model)
    bank=DataBank(cfg);out=Path(args.output)
    assert not (out/'summary.json').exists(),'Do not repeat an already completed frozen evaluation'
    out.mkdir(parents=True,exist_ok=True)
    original=policy.pil;policy.pil=lambda rgb:demonstration_transport(original(rgb))
    atomic(out/'experiment.json',{'time':time.time(),'pid':os.getpid(),'checkpoint':str(checkpoint),'commit':marker,
        'treatment':'228x228 bilinear resize and JPEG92 exactly as successful demonstration collection',
        'baseline':str(root/f'evaluations/step_{args.step:06d}/development/summary.json'),
        'split':'development','training':False,'fixed_split_unchanged':True,'horizon':32,
        'all_actions_from_discrete_symbolic_rules':True})
    simulators=[bank.simulator(out/f'sim_{i}.log',0) for i in range(3)]
    rows=[]
    try:
        for offset in range(0,len(bank.dev),3):
            episodes=bank.dev[offset:offset+3]
            policies,metrics=policy.rollout_many(model,simulators[:len(episodes)],episodes,stochastic=False)
            for ep,p,result in zip(episodes,policies,metrics):
                save_trace(out/ep['episode_id'],p,ep,result)
                steps=[s for c in p.chunks for s in c['rows']]
                rows.append({k:ep[k] for k in ['episode_id','scene','domain','category','distance_group']}|
                    {'success':bool(result.get('success')),'steps':len(steps),'collisions':sum(s['collision'] for s in steps)})
            atomic(out/'progress.json',{'time':time.time(),'pid':os.getpid(),'completed':len(rows),
                'successes':sum(r['success'] for r in rows)})
    finally:
        for sim in simulators:sim.close()
    atomic(out/'episodes.json',rows);summary=aggregate(rows);atomic(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
