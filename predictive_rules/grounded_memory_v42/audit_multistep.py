"""Training-only multi-step calibration under an immutable v42 checkpoint.

Existing recorded actions are replayed in generated symbolic tables, not in a
simulator. Grounded labels are used only by the scorer. Neither predictive
memory nor imagined observations are replaced by those labels in open-loop
forecasting. The allowed-feedback comparison refreshes observation IDs only.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from data import ROOT,sha
from model import FactoredWorld,Config,hard
from planner import Program


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--update',type=int,required=True);args=parser.parse_args()
    torch.set_num_threads(3)
    run=ROOT/'run_grounded_memory_v42';folder=run/'diagnostics';folder.mkdir(exist_ok=True)
    checkpoint=run/f'checkpoint_{args.update:06d}.pt'
    commit=json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
    pin=folder/f'multistep_{os.getpid()}.pt';os.link(checkpoint,pin)
    try:
        assert sha(pin)==commit['sha256']
        saved=torch.load(pin,map_location='cpu',weights_only=False)
        model=FactoredWorld(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model']);c=model.cfg
        cache=ROOT/'onpolicy_successors_v40'
        groups=defaultdict(list)
        for p in sorted((cache/'counterfactual_onpolicy').glob('*.json')):
            r=json.loads(p.read_text());assert r['training_only'] and not r['goal_supervision']
            if r['executed_actions']>=5:groups[(r['domain'],r['category'],r['distance_group'])].append((p,r))
        # Up to three per group, deterministically selected without outcomes.
        chosen=[pair for key in sorted(groups) for pair in groups[key][:3]]
        rows=[];began=time.time()
        for p,r in chosen:
            with np.load(p.with_suffix('.npz')) as archive:
                x=torch.from_numpy(archive['features'])[None]
                actions=archive['recorded_actions']
            previous=torch.tensor([[17]+actions[:x.shape[1]-1].tolist()])
            lo,z,_=model.observation_encoding(x);ob=hard(lo)
            lm,_,_=model.memory_sequence(ob,previous,z)
            memory=lm.argmax(-1)[0].numpy();observed=lo.argmax(-1)[0].numpy()
            task=torch.tensor([0 if r['category']=='chair' else 1])
            context=model.context_sequence(x,ob,previous,task)
            # One initial and one later boundary; labels never select a root.
            for start in (0,5):
                length=min(10,len(actions)-start,len(observed)-start-1)
                if length<1:continue
                exported=model.compile(context[:,start],task);program=Program(**exported).validate()
                initial=int(memory[start])*c.observations+int(observed[start])
                distribution=np.zeros(c.memories*c.observations,np.float64);distribution[initial]=1.
                modal_m=int(memory[start]);modal_o=int(observed[start]);feedback_m=modal_m
                for k in range(1,length+1):
                    t=start+k-1;action=int(actions[t])
                    future=program.next_memory[:,action]
                    weighted=distribution[:,None]*program.outcome_probability[:,action,:]
                    next_distribution=np.zeros((c.memories,c.observations),np.float64)
                    np.add.at(next_distribution,future,weighted)
                    distribution=next_distribution.ravel()
                    assert np.isclose(distribution.sum(),1.,atol=2e-5)
                    old=modal_m*c.observations+modal_o
                    modal_m=int(program.next_memory[old,action]);modal_o=int(program.outcome_probability[old,action].argmax())
                    known_o=int(observed[t]);old_feedback=feedback_m*c.observations+known_o
                    feedback_o=int(program.outcome_probability[old_feedback,action].argmax())
                    feedback_m=int(program.next_memory[old_feedback,action])
                    tm,to=int(memory[t+1]),int(observed[t+1]);truth=tm*c.observations+to
                    po=distribution.reshape(c.memories,c.observations).sum(0)
                    rows.append(dict(episode=r['episode'],start=start,horizon=k,
                        modal_memory_correct=modal_m==tm,modal_observation_correct=modal_o==to,
                        modal_joint_correct=modal_m==tm and modal_o==to,
                        probability_map_joint_correct=int(distribution.argmax())==truth,
                        probability_map_observation_correct=int(po.argmax())==to,
                        true_joint_probability=float(distribution[truth]),true_observation_probability=float(po[to]),
                        impossible_true_joint=bool(distribution[truth]==0),
                        observation_feedback_memory_correct=feedback_m==tm,
                        observation_feedback_observation_correct=feedback_o==to,
                        unchanged_initial_observation_baseline=int(observed[start])==to))
            print(json.dumps(dict(completed_episodes=len({x['episode'] for x in rows}),seconds=time.time()-began)),flush=True)
        by_horizon={}
        boolean_fields=[k for k,v in rows[0].items() if isinstance(v,(bool,np.bool_))]
        for k in range(1,11):
            selected=[r for r in rows if r['horizon']==k]
            if not selected:continue
            value={key:sum(bool(r[key]) for r in selected)/len(selected) for key in boolean_fields}
            value.update(samples=len(selected),
                joint_nll=sum(-math.log(max(r['true_joint_probability'],1e-12)) for r in selected)/len(selected),
                observation_nll=sum(-math.log(max(r['true_observation_probability'],1e-12)) for r in selected)/len(selected))
            by_horizon[k]=value
        result=dict(training_only=True,source_commit=commit,episodes=len(chosen),roots=len({(r['episode'],r['start']) for r in rows}),
            seconds=time.time()-began,by_horizon=by_horizon,
            interpretation='Recorded training trajectories, not a navigation success-rate experiment. Probability propagation retains every generated symbolic outcome. Observation-feedback comparison never corrects memory using future evidence.',
            rows=rows)
        (folder/f'multistep_{args.update:06d}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    finally:
        if pin.exists():pin.unlink()


if __name__=='__main__':main()
