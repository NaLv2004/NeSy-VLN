"""Saved-proof diagnostic only. Does not extend or rerun any evaluation."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from hard_interpreter import Program,plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder');ap.add_argument('--out',required=True);args=ap.parse_args()
    counts=Counter();ages=Counter();failure_steps=Counter();stops=[];examples=[]
    for path in sorted(Path(args.folder).glob('*/trace.json')):
        r=json.loads(path.read_text());programs={}
        for event in r['programs']:
            with np.load(path.parent/event['file'],allow_pickle=False) as d:
                p=Program(d['next_state'],d['confidence'],d['correction'],int(d['goal']),int(d['start'])).validate()
            programs[p.digest()]=p
        for step in r['steps']:
            p=programs[step['program_sha256']];s=step['proof']['corrected']
            counts['observed_stop_rows']+=1;counts['stop_keeps_current_symbol']+=int(p.next_state[s,0])==s
        if r['planning_failed']:
            f=r['failed_proof'];p=programs[f['program_sha256']];s=len(r['steps']);failure_steps[s]+=1;ages[s-p.start]+=1
            extended=plan(p,f['predicted'],f['observed'],horizon=10)
            counts['planning_failures']+=1
            if extended['found']:
                counts['failure_only_under_remaining_episode_budget']+=1
            else:
                counts['unreachable_even_with_full_ten_step_proof']+=1
                if s==0:counts['initial_unreachable']+=1
                if s==p.start:counts['unreachable_immediately_after_generation']+=1
            examples.append(dict(episode=r['episode']['episode_id'],step=s,program_age=s-p.start,
                                 horizon=f['horizon'],reachable_with_ten_steps=extended['found'],metrics=r['metrics']))
        if r['steps'] and r['steps'][-1]['action']==0:
            x=r['steps'][-1];proof=x['proof']
            stops.append(dict(episode=r['episode']['episode_id'],success=r['metrics'].get('success'),
                              source=proof['corrected'],goal=proof['goal'],stop_changed_symbol=proof['states'][0]!=proof['states'][1],
                              metrics=r['metrics']))
    result=dict(evaluation=str(args.folder),saved_programs_only=True,no_new_rollouts=True,counts=dict(counts),
                failure_steps=dict(sorted(failure_steps.items())),program_age_at_failure=dict(sorted(ages.items())),
                stops=stops,failed_episodes=examples)
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['stops','failed_episodes']}))
    print('STOP_SUMMARY',json.dumps(dict(total=len(stops),false_stops=sum(not x['success'] for x in stops),
        changed_symbol_false_stops=sum(not x['success'] and x['stop_changed_symbol'] for x in stops))))

if __name__=='__main__':main()
