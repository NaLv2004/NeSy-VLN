"""CPU-only training-corpus diagnosis of grounding, prediction and search.

No navigation evaluation is repeated. No held-out data is loaded or optimized.
"""
import argparse,json,random,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from predictive_nesy.data import load_training_windows,batch_windows
from predictive_nesy.model_memory import PredictiveMemoryModel,MemoryConfig,scores
from predictive_nesy.model import straight_through
from predictive_nesy.interpreter import export_program
from predictive_nesy import belief_interpreter as logic


def path_score(program,state,actions):
    total=0.;stopped=False
    for i,act in enumerate(actions):
        if act==0:
            total+=.97**i*(2*logic.scalar(program,state,'goals')-1);stopped=True;break
        total+=.97**i*(-.02-.2*logic.risk_all(program,state)[act])
        state=logic.successors(program,state)[act]
    if not stopped:total+=.97**len(actions)*logic.scalar(program,state,'values')
    return float(total)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--windows',type=int,default=64);args=p.parse_args()
    torch.set_num_threads(2);begin=time.time()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=PredictiveMemoryModel(MemoryConfig(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
    m=saved['manifest'];windows,provenance=load_training_windows(m['data']['source'],m['data']['train_layouts'])
    assert m['data']==provenance
    labels=np.load(Path(args.checkpoint).parent.parent/'run_v1/learned_symbol_targets.npz')['labels']
    for w,y in zip(windows,labels):w['teacher_ids']=y.astype(np.int64)
    rng=random.Random(1520);candidates=[w for w in windows if w['valid'].sum()>=4]
    chosen=rng.sample(candidates,min(len(candidates),args.windows))
    rows=[];curves=defaultdict(list)
    for w in chosen:
        b=batch_windows([w],'cpu');start=w['start'];width=min(start+1,11)
        program=model.generate(b['task'],b['history'][:,:width],b['previous_actions'][:,:width],b['history_length'],start=start)
        observed=model.training_targets(program,b['targets'],b['actions'])
        initial=straight_through(observed[:,0]);q=scores(program,initial,model.pairs)[0]
        exported=export_program(program);state=initial[0].numpy();teacher_actions=w['actions'][:min(int(w['valid'].sum()),10)].tolist()
        row=dict(episode_id=w['episode_id'],start=start,first_teacher_action=int(w['actions'][0]),
            one_step_action=int(q.argmax()),teacher_length=len(teacher_actions),plans={})
        for horizon in [1,3,5,10]:
            plan=logic.plan(exported,state,horizon=horizon)
            row['plans'][str(horizon)]=dict(actions=plan['actions'],score=plan['score'])
        row['teacher_score']=path_score(exported,state,teacher_actions)
        for t in range(int(w['valid'].sum())):
            actual=observed[0,t].argmax(-1).numpy()
            goal=float(logic.scalar(exported,state,'goals'))
            actual_goal=float(logic.scalar(exported,np.eye(16,dtype=np.float32)[actual],'goals'))
            curves[t].append(dict(observed_state_accuracy=float((state[:4].argmax(-1)==w['teacher_ids'][t]).mean()),
                memory_state_accuracy=float((state[4:].argmax(-1)==actual[4:]).mean()),
                entropy=float(-(state*np.log(np.maximum(state,1e-8))).sum(-1).mean()),
                goal=goal,recognized_goal=actual_goal,true_goal=bool(w['goal'][t]),
                goal_brier=(goal-float(w['goal'][t]))**2))
            if int(w['actions'][t])==0:break
            state=logic.successors(exported,state)[int(w['actions'][t])]
        rows.append(row)
    def mean(values):return float(np.mean(values)) if values else None
    summary=dict(checkpoint=args.checkpoint,update=saved['update'],training_only=True,windows=len(rows),
        one_step_first_action_accuracy=mean([r['one_step_action']==r['first_teacher_action'] for r in rows]),
        by_horizon={h:dict(first_action_accuracy=mean([r['plans'][h]['actions'][0]==r['first_teacher_action'] for r in rows]),
            differs_from_one_step=mean([r['plans'][h]['actions'][0]!=r['one_step_action'] for r in rows]),
            immediate_stop_rate=mean([r['plans'][h]['actions'][0]==0 for r in rows])) for h in ['1','3','5','10']},
        time_curve={t:dict(samples=len(v),**{k:mean([r[k] for r in v]) for k in v[0]}) for t,v in curves.items()},
        demonstration_stop=dict(samples=sum(r['true_goal'] for v in curves.values() for r in v),
            predicted_goal=mean([r['goal'] for v in curves.values() for r in v if r['true_goal']]),
            recognized_goal=mean([r['recognized_goal'] for v in curves.values() for r in v if r['true_goal']])),
        seconds=time.time()-begin)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'rows.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
