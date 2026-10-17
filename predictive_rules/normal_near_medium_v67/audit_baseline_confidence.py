"""Offline probability calibration audit of completed baseline; zero rollouts."""
import collections,json,math,statistics,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_normal_near_medium_v67';SOURCE=OUT/'baseline_v63_dev80'
files=sorted(SOURCE.glob('*/trace.json'));assert len(files)==80
episodes=[];transitions=[];all_steps=[]
for p in files:
    r=json.loads(p.read_text());success=bool(r['metrics'].get('success',False));programs=r['programs'];steps=r['steps']
    e=dict(episode=r['episode']['episode_id'],success=success,distance_group=r['episode']['distance_group'],
        category=r['episode']['category'],steps=len(steps),failed_horizon=r.get('failed_proof',{}).get('horizon'),
        false_stop=bool(steps and steps[-1]['action']==0 and not success),
        initial_probability=steps[0]['proof']['model_goal_probability'] if steps else 0,
        initial_failure_probability=steps[0]['proof']['model_failure_probability'] if steps else 1,
        no_preimage_programs=sum(not v['terminal_preimage'] for v in programs),
        ever_goal_observed=False,initial_distance=r['episode']['initial_distance_m'],metrics=r['metrics'])
    for j,s in enumerate(steps):
        program=max((q for q in programs if q['start']<=s['step']),key=lambda q:q['start'])
        e['ever_goal_observed']|=s['observed']==program['goal']%64
        all_steps.append(dict(success=success,action=s['action'],probability=s['proof']['model_goal_probability'],
            failure=s['proof']['model_failure_probability']))
        if j+1<len(steps):
            actual=steps[j+1]['observed_before_renewal'];probs=s['proof']['root_outcome_probabilities'];modal=max(range(64),key=lambda i:probs[i])
            transitions.append(dict(hit=modal==actual,p_actual=probs[actual],p_max=probs[modal],
                nll=-math.log(max(1e-30,probs[actual])),entropy=-sum(p*math.log(p) for p in probs if p>0)))
    episodes.append(e)
def mean(xs):return statistics.mean(xs) if xs else None
def stats(rows):
    return dict(episodes=len(rows),successes=sum(e['success'] for e in rows),mean_initial_predicted_success=mean([e['initial_probability'] for e in rows]),
        goal_observed_episodes=sum(e['ever_goal_observed'] for e in rows),false_stops=sum(e['false_stop'] for e in rows),
        no_terminal_preimage_episodes=sum(e['no_preimage_programs']>0 for e in rows))
result=dict(time=time.time(),new_rollouts=0,optimizer_updates=0,source=str(SOURCE),
    overall=stats(episodes),by_outcome={str(k):stats([e for e in episodes if e['success']==k]) for k in (False,True)},
    first_probability_bins={str(t):stats([e for e in episodes if e['initial_probability']>=t]) for t in (.5,.9,.99,.999)},
    transition_count=len(transitions),prediction_accuracy=mean([t['hit'] for t in transitions]),
    mean_transition_nll=mean([t['nll'] for t in transitions]),mean_modal_probability=mean([t['p_max'] for t in transitions]),
    mean_realized_probability=mean([t['p_actual'] for t in transitions]),mean_entropy=mean([t['entropy'] for t in transitions]),
    modal_calibration={f'{a}-{b}':dict(count=len(v),accuracy=mean([x['hit'] for x in v]),confidence=mean([x['p_max'] for x in v]))
        for a,b in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.001)] if (v:=[t for t in transitions if a<=t['p_max']<b])},
    action_histogram=dict(collections.Counter(s['action'] for s in all_steps)),episode_rows=episodes,
    interpretation_limit='Development-set diagnostics, not independent final-test evidence. Symbolic model probability is not calibrated physical success probability.')
(OUT/'baseline_probability_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='episode_rows'},indent=2))
