"""CPU-only audit of already completed full dev80 runs. No new rollouts."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')

def mean(values):
    return sum(values) / len(values) if values else None

def analyse(folder):
    summary = json.loads((folder / 'summary.json').read_text())
    traces = [json.loads(p.read_text()) for p in sorted(folder.glob('*/trace.json'))]
    assert len(traces) == summary['episodes'] == 80
    assert len({r['episode']['episode_id'] for r in traces}) == 80
    transitions = []; episodes = []; groups = defaultdict(list)
    for trace in traces:
        steps = trace['steps']; episode = trace['episode']; starts = [p['start'] for p in trace['programs']]
        joint_outcomes = defaultdict(Counter)
        for current, following in zip(steps, steps[1:]):
            p = current['proof']['root_outcome_probabilities']; total = sum(p); p = [v / total for v in p]
            target = following['observed_before_renewal']
            program_start = max(s for s in starts if s <= current['step'])
            key = (program_start, current['memory'], current['observed'], current['action'])
            joint_outcomes[key][target] += 1
            prediction = max(range(len(p)), key=p.__getitem__)
            transitions.append(dict(episode=episode['episode_id'], step=current['step'],
                domain=episode['domain'], age=current['step']-program_start,
                correct=prediction == target, persistence=current['observed'] == target,
                true_probability=p[target], confidence=max(p),
                entropy=-sum(v*math.log(v) for v in p if v),
                collision=current.get('collision_for_report_only',False),
                goal_probability=current['proof']['model_goal_probability'],
                action=current['action']))
        repeated = [counts for counts in joint_outcomes.values() if sum(counts.values()) > 1]
        success = bool(trace['metrics'].get('success',False))
        initial = steps[0]['proof']['model_goal_probability'] if steps else None
        last = steps[-1] if steps else {}
        item=dict(episode=episode['episode_id'], success=success, steps=len(steps),
            initial_goal_probability=initial,
            false_stop=bool(last.get('action') == 0 and not success),
            unique_actions=len({s['action'] for s in steps}),
            repeated_state_action_rows=len(repeated),
            repeated_row_observations=sum(sum(c.values()) for c in repeated),
            repeated_row_modal_correct=sum(max(c.values()) for c in repeated),
            conflicting_state_action_rows=sum(len(c)>1 for c in repeated))
        episodes.append(item)
        for key in ('domain','distance_group','category'):
            groups[key+':'+episode[key]].append(item)

    def stats(rows):
        return dict(n=len(rows),**{key:mean([r[key] for r in rows]) for key in
            ('correct','persistence','true_probability','confidence','entropy','collision','goal_probability')})
    def estats(rows):
        initial=[r['initial_goal_probability'] for r in rows if r['initial_goal_probability'] is not None]
        return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
            false_stops=sum(r['false_stop'] for r in rows),
            mean_initial_goal_probability=mean(initial),
            mean_steps=mean([r['steps'] for r in rows]),
            repeated_state_action_rows=sum(r['repeated_state_action_rows'] for r in rows),
            conflicting_state_action_rows=sum(r['conflicting_state_action_rows'] for r in rows),
            empirical_repeated_row_ceiling=(sum(r['repeated_row_modal_correct'] for r in rows)/sum(r['repeated_row_observations'] for r in rows)
                if sum(r['repeated_row_observations'] for r in rows) else None))
    return dict(path=str(folder),summary=summary,episodes=estats(episodes),transitions=stats(transitions),
        initial_transitions=stats([r for r in transitions if r['step']==0]),
        by_age={str(age):stats([r for r in transitions if r['age']==age]) for age in range(6)},
        by_confidence={str((lo,hi)):stats([r for r in transitions if lo<=r['confidence']<hi])
            for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,.9),(.9,1.000001))},
        changed_observations=stats([r for r in transitions if not r['persistence']]),
        colliding_transitions=stats([r for r in transitions if r['collision']]),
        noncolliding_transitions=stats([r for r in transitions if not r['collision']]),
        by_group={key:estats(rows) for key,rows in groups.items()},
        success_only=estats([r for r in episodes if r['success']]),
        failure_only=estats([r for r in episodes if not r['success']]),
        episode_ids=sorted(r['episode'] for r in episodes),
        limitations=['Observed repeated-row ceiling is only an in-sample lower bound on aliasing; unique rows do not prove Markov sufficiency.',
            'Each checkpoint chooses different trajectories, so per-transition differences are descriptive, not matched-action causal comparisons.',
            'Learned goal probability measures symbolic goal reachability, not a calibrated physical success probability.'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folders',nargs='+',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();folders=[ROOT/f for f in args.folders]
    assert all(f.is_relative_to(ROOT) for f in folders)
    runs=[analyse(folder) for folder in folders]
    assert all(r['episode_ids']==runs[0]['episode_ids'] for r in runs)
    result=dict(time=time.time(),new_simulator_rollouts=0,gradients=0,device='CPU',finalfixed_used=False,runs=runs)
    destination=Path(args.out);assert destination.is_relative_to(ROOT)
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(path=str(destination),runs=[{k:r[k] for k in ('path','episodes','transitions','initial_transitions','changed_observations')} for r in runs])))
