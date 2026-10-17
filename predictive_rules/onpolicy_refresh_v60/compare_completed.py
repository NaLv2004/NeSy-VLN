"""Compare already completed full80 trajectories without re-running policies."""
import json,math,statistics,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_onpolicy_refresh_v60'
sources={'v57_1000':ROOT/'run_goal_equivalence_v57/dev_001000',
         'v60_500':ROOT/'run_onpolicy_refresh_v60/dev_000500',
         'v60_1000':ROOT/'run_onpolicy_refresh_v60/dev_001000'}
all_rows={};summary={}
for name,folder in sources.items():
    rows={};actual_probability=[];confidence=[];correct=[];changed_correct=[]
    eligible=[]
    for p in sorted(folder.glob('*/trace.json')):
        r=json.loads(p.read_text());e=r['episode'];steps=r['steps']
        rows[e['episode_id']]=dict(success=bool(r['metrics'].get('success')),domain=e['domain'],
            distance=e['distance_group'],category=e['category'],steps=len(steps),metrics=r['metrics'])
        for a,b in zip(steps,steps[1:]):
            prob=a['proof']['root_outcome_probabilities'];total=sum(prob)
            target=b['observed_before_renewal'];actual_probability.append(prob[target]/total)
            confidence.append(max(prob)/total);correct.append(a['predicted_observation']==target)
            if a['observed']!=target:changed_correct.append(a['predicted_observation']==target)
    assert len(rows)==80
    all_rows[name]=rows
    summary[name]=dict(successes=sum(r['success'] for r in rows.values()),episodes=80,
        prediction_accuracy=statistics.mean(correct),changed_accuracy=statistics.mean(changed_correct),
        prediction_mean_confidence=statistics.mean(confidence),
        nll=statistics.mean(-math.log(max(p,1e-300)) for p in actual_probability),
        actual_outcome_probability_mean=statistics.mean(actual_probability),
        actual_outcome_probability_median=statistics.median(actual_probability),
        actual_outcome_probability_under_001=statistics.mean(p<.01 for p in actual_probability))
baseline=all_rows['v57_1000'];comparisons={}
for name,rows in all_rows.items():
    assert set(rows)==set(baseline)
    if name=='v57_1000':continue
    comparisons[name]=dict(gained=[e for e,r in rows.items() if r['success'] and not baseline[e]['success']],
        lost=[e for e,r in rows.items() if not r['success'] and baseline[e]['success']],
        both_success=sum(r['success'] and baseline[e]['success'] for e,r in rows.items()))
result=dict(time=time.time(),summary=summary,paired=comparisons,episodes=all_rows,
    same_original_full80=True,new_rollouts=0,finalfixed80_used=False,
    limitation='One deterministic evaluation per checkpoint; paired counts describe changes and do not prove statistical or causal improvement.')
(OUT/'completed_comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(summary=summary,paired={k:{'gained':len(v['gained']),'lost':len(v['lost']),'both_success':v['both_success']} for k,v in comparisons.items()})))
