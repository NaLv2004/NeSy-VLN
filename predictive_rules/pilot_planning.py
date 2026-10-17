"""Bounded CPU optimization check of multi-step ranking, using training data only."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.data import load_training_windows,batch_windows,sha
from predictive_nesy.model_memory import PredictiveMemoryModel,MemoryConfig
from predictive_nesy.training_memory import objective
from predictive_nesy.planning_alignment import loss as plan_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--updates',type=int,default=64);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(1522);rng=random.Random(1522)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);m=PredictiveMemoryModel(MemoryConfig(**saved['model_config']));m.load_state_dict(saved['model'])
    manifest=saved['manifest'];windows,provenance=load_training_windows(manifest['data']['source'],manifest['data']['train_layouts']);assert provenance==manifest['data']
    labels=np.load(Path(args.checkpoint).parent.parent/'run_v1/learned_symbol_targets.npz')['labels']
    for w,y in zip(windows,labels):w['teacher_ids']=y.astype(np.int64)
    chosen=rng.sample([w for w in windows if w['start']==0 and w['valid'].sum()>=4 and w['supervised'][0]],48)
    training,unseen=chosen[:32],chosen[32:]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    def measure(data):
        rows=[];m.eval()
        with torch.no_grad():
            for i in range(0,len(data),4):
                batch=batch_windows(data[i:i+4],'cpu');_,a=objective(m,batch);_,b=plan_loss(m,batch,beam_width=32);rows.append(dict(**a,**b))
        return {k:float(np.mean([r[k] for r in rows if isinstance(r.get(k),(int,float))]))
                for k in rows[0] if any(isinstance(r.get(k),(int,float)) for r in rows)}
    begin=time.time();before={'optimization_subset':measure(training),'other_training_windows':measure(unseen)}
    print(json.dumps(dict(stage='before',metrics=before)),flush=True)
    optim=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
    for step in range(args.updates):
        m.train();b=batch_windows(rng.choices(training,k=4),'cpu');optim.zero_grad(set_to_none=True)
        main_loss,metrics=objective(m,b);main_loss.backward()
        alignment,extra=plan_loss(m,b,beam_width=32);(5*alignment).backward()
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite pilot gradient')
        optim.step()
        if (step+1)%16==0:print(json.dumps(dict(update=step+1,**extra)),flush=True)
    after={'optimization_subset':measure(training),'other_training_windows':measure(unseen)}
    report=dict(checkpoint=args.checkpoint,checkpoint_sha256=sha(args.checkpoint),updates=args.updates,
        training_only=True,device='cpu',navigation_evaluations=0,before=before,after=after,seconds=time.time()-begin,
        training_ids=[w['episode_id'] for w in training],diagnostic_ids=[w['episode_id'] for w in unseen])
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
