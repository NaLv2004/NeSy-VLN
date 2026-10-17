"""Check whether the old learned observation alphabet admits deterministic dynamics."""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from predictive_nesy.data import load_training_windows,read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');a=parser.parse_args()
    run=Path(a.run);manifest=read(run/'manifest.json')
    windows,provenance=load_training_windows(manifest['data']['source'],manifest['data']['train_layouts'])
    labels=np.load(run/'learned_symbol_targets.npz')['labels']
    assert len(labels)==len(windows)
    counts=Counter();examples=[]
    for window,states in zip(windows,labels):
        trans=defaultdict(list);goals=defaultdict(set)
        n=int(window['valid'].sum())
        for t in range(n):
            goals[tuple(states[t])].add(bool(window['goal'][t]))
            if t+1<n and window['actions'][t]!=0:
                source=tuple(states[t]);target=tuple(states[t+1]);act=int(window['actions'][t])
                trans[(source,act)].append((t,target))
                counts['transitions']+=1;counts['identical_successor']+=source==target
        conflicts={str(key):[(t,list(map(int,target))) for t,target in rows] for key,rows in trans.items() if len({target for _,target in rows})>1}
        contradictory_goal=any(len(values)>1 for values in goals.values())
        counts['windows']+=1;counts['dynamics_conflict_windows']+=bool(conflicts)
        counts['goal_alias_windows']+=contradictory_goal
        if conflicts and len(examples)<8:
            examples.append(dict(episode=window['episode_id'],start=window['start'],conflicts=conflicts,goal_alias=contradictory_goal))
    report=dict(counts,source_alphabet='Old observation IDs used by the new v1 supervision',
        identical_successor_rate=counts['identical_successor']/counts['transitions'],
        dynamics_conflict_window_rate=counts['dynamics_conflict_windows']/counts['windows'],
        goal_alias_window_rate=counts['goal_alias_windows']/counts['windows'],examples=examples)
    (run/'state_aliasing_diagnosis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
