"""Read completed traces only. No neural forward, simulator, training or eval."""
import json,math,statistics,time
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sources=dict(v46_legacy=ROOT/'diagnostic_discounted_rules_v46/baseline',
    v50_stable=ROOT/'diagnostic_stable_probability_v50/physical',
    v51_maximum_path=ROOT/'diagnostic_path_rollout_v51/physical',
    v48_best500=ROOT/'run_contextual_symbols_v48/dev_000500',
    v48_final3000=ROOT/'run_contextual_symbols_v48/dev_003000')

def mean(values):return statistics.fmean(values) if values else None
def summarize(rows):
    return dict(n=len(rows),accuracy=mean([r['correct'] for r in rows]),
        mean_confidence=mean([r['confidence'] for r in rows]),
        actual_outcome_probability_mean=mean([r['actual_probability'] for r in rows]),
        actual_outcome_probability_median=statistics.median([r['actual_probability'] for r in rows]) if rows else None,
        negative_log_likelihood=mean([-math.log(max(1e-300,r['actual_probability'])) for r in rows]),
        actual_probability_below_001=mean([r['actual_probability']<.01 for r in rows]),
        actual_probability_below_1e6=mean([r['actual_probability']<1e-6 for r in rows]),
        collision_rate=mean([r['collision'] for r in rows]),
        observation_persistence=mean([r['persistence'] for r in rows]))

report={}
for label,folder in sources.items():
    files=sorted(folder.glob('*/trace.json'));assert len(files)==(80 if label.startswith('v48') else 24)
    pairs=[];episodes=[]
    for path in files:
        trace=json.loads(path.read_text());steps=trace['steps'];age=0
        for current,following in zip(steps,steps[1:]):
            if current['renewed']:age=0
            p=current['proof']['root_outcome_probabilities'];mass=sum(p);p=[x/mass for x in p]
            actual=following.get('observed_before_renewal',following['observed'])
            pairs.append(dict(correct=current['predicted_observation']==actual,confidence=max(p),actual_probability=p[actual],
                action=current['action'],collision=current['collision_for_report_only'],
                persistence=current['observed']==actual,age=age,step=current['step'],episode=trace['episode']['episode_id']))
            age+=1
        first=steps[0]['proof'] if steps else trace.get('failed_proof',{})
        episodes.append(dict(episode=trace['episode']['episode_id'],success=bool(trace['metrics'].get('success')),
            initial_goal_score=first.get('model_goal_probability'),initial_path_score=first.get('maximum_path_probability'),
            last_metrics=trace['metrics'],steps=len(steps),planning_failed=trace.get('planning_failed',False)))
    bins={}
    for low,high in [(0,.25),(.25,.5),(.5,.75),(.75,.9),(.9,1.00001)]:
        bins[f'{low}-{high}']=summarize([r for r in pairs if low<=r['confidence']<high])
    starts=[r['initial_goal_score'] for r in episodes if r['initial_goal_score'] is not None]
    report[label]=dict(episodes=len(episodes),successes=sum(r['success'] for r in episodes),
        initial_goal_score_mean=mean(starts),initial_goal_score_median=statistics.median(starts) if starts else None,
        transition_summary=summarize(pairs),confidence_bins=bins,
        program_age={str(a):summarize([r for r in pairs if r['age']==a]) for a in range(6)},
        initial_transitions=summarize([r for r in pairs if r['step']==0]),
        episodes_details=episodes,by_action={str(a):summarize([r for r in pairs if r['action']==a]) for a in range(17)})
out=ROOT/'diagnostic_saved_reliability_v52';out.mkdir(exist_ok=True)
result=dict(time=time.time(),sources={k:str(v) for k,v in sources.items()},results=report,
    no_new_rollouts=True,no_training_data_modified=True,no_policy_access_to_scoring_metrics=True,
    limitations='Paired diagnostic variants reuse24training tasks; development traces are existing scheduled evaluations. Confidence calibration is descriptive on policy-selected actions, not independent causal attribution.')
(out/'report.json').write_text(json.dumps(result,indent=2))
print(json.dumps({label:{k:v for k,v in item.items() if k not in ('episodes_details','by_action','program_age','confidence_bins')} for label,item in report.items()}))
