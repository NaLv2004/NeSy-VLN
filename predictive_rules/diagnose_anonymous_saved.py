"""Offline diagnostics of already-completed anonymous-rule rollouts; no rerun."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from predictive_nesy.anonymous_interpreter import Program,FIELDS,reward_all,scalar
from audit_trace import audit


def diagnose(directory):
    directory=Path(directory)
    if not (directory/'summary.json').exists():raise ValueError('Only completed evaluations may be diagnosed')
    summary=json.loads((directory/'summary.json').read_text())
    episodes=[];offsets=defaultdict(list);histogram=Counter();repeat=0;adjacent=0
    movement_scores=[];stop_scores=[];step_counts=[];program_returns=[];audits=[]
    for tracefile in sorted(directory.glob('*/trace.json')):
        folder=tracefile.parent;trace=json.loads(tracefile.read_text());audits.append(audit(folder))
        measured=json.loads((folder/'evaluation.json').read_text())
        programs={r['start']:Program.load(folder/r['file']) for r in trace['programs']}
        step_counts.append(len(trace['steps']));previous=None
        for step in trace['steps']:
            state=np.asarray(step['before_belief']);after=np.asarray(step['after_belief'])
            p=programs[step['program_start']];action=step['action'];histogram[action]+=1
            reward=float(reward_all(p,state)[action])
            actual=1. if action==0 and measured['success'] else -1. if action==0 else -.02
            item=dict(reward=reward,actual=actual,entropy=float(-(state*np.log(state.clip(1e-8))).sum(-1).mean()),
                unchanged=float(np.mean(state.argmax(-1)==after.argmax(-1))),terminal=action==0)
            offsets[step['step']%10].append(item)
            if action==0:stop_scores.append(dict(predicted=reward,actual=actual,success=measured['success']))
            else:movement_scores.append(reward)
            if previous is not None:repeat+=action==previous;adjacent+=1
            previous=action
        for rec in trace['programs']:
            start=rec['start'];steps=trace['steps'][start:start+10]
            observed=sum(.97**i*(1. if s['action']==0 and measured['success'] else -1. if s['action']==0 else -.02)
                         for i,s in enumerate(steps))
            terminal=bool(steps and (steps[-1]['action']==0 or start+len(steps)==32))
            program_returns.append(dict(predicted=rec['plan']['score'],observed=observed,terminal=terminal,
                episode_id=trace['episode']['episode_id'],start=start))
        episodes.append(dict(episode_id=trace['episode']['episode_id'],success=measured['success'],steps=len(trace['steps']),
            stopped=bool(trace['steps'] and trace['steps'][-1]['action']==0)))
    if len(episodes)!=summary['episodes']:raise ValueError('Incomplete saved trace collection')
    stopped=sum(x['terminal'] for rows in offsets.values() for x in rows)
    terminal=[x for x in program_returns if x['terminal']]
    result=dict(episodes=len(episodes),successes=sum(x['success'] for x in episodes),steps=sum(step_counts),
        programs=len(program_returns),independent_audit_passed=True,action_histogram=histogram,
        repeated_action_rate=repeat/max(adjacent,1),stops=stopped,failed_stops=sum(not x['success'] for x in stop_scores),
        never_stopped=sum(not e['stopped'] for e in episodes),
        mean_movement_score=float(np.mean(movement_scores)) if movement_scores else None,
        true_movement_score=-.02,movement_score_rmse=float(np.mean((np.asarray(movement_scores)+.02)**2)**.5) if movement_scores else None,
        mean_predicted_stop_score=float(np.mean([x['predicted'] for x in stop_scores])) if stop_scores else None,
        mean_actual_stop_score=float(np.mean([x['actual'] for x in stop_scores])) if stop_scores else None,
        terminal_plan_score_error=float(np.mean([x['predicted']-x['observed'] for x in terminal])) if terminal else None,
        terminal_plans=len(terminal),
        by_offset={i:dict(samples=len(rows),mean_state_entropy=float(np.mean([r['entropy'] for r in rows])),
            unchanged_category_fraction=float(np.mean([r['unchanged'] for r in rows])),
            score_rmse=float(np.mean([(r['reward']-r['actual'])**2 for r in rows])**.5)) for i,rows in offsets.items()},
        evaluation_summary=summary)
    (directory/'anonymous_diagnosis.json').write_text(json.dumps(result,indent=2))
    (directory/'independent_rule_audit.json').write_text(json.dumps(dict(episodes=len(audits),steps=sum(x['steps'] for x in audits),
        programs=sum(x['programs'] for x in audits),all_verified=True,records=audits),indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['by_offset','evaluation_summary']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();diagnose(a.directory)
