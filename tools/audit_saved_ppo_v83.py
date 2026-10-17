"""Inspect completed PPO evidence only; no model inference or new navigation."""
import json, math, time, statistics
from collections import Counter, defaultdict
from pathlib import Path

B=Path('/home/user/xjy/nesy_vla_20261010')
P=B/'perceptual_rl_v80/joint'
read=lambda p:json.loads(p.read_text())
assert read(P/'status.json')['state']=='complete'
def summarize(paths,training=False):
    counts=Counter();terms=defaultdict(float);groups=defaultdict(Counter);outcomes=defaultdict(list)
    factors=[Counter() for _ in range(4)];memory=[Counter() for _ in range(4)];episode_ids=[]
    maximum_reward_error=0.;failure_distances=[]
    for path in paths:
        t=read(path);steps=t['steps'];e=t['episode'];success=bool(t['metrics']['success'])
        outcome='success' if success else ('false_stop' if steps[-1]['action']==0 else 'horizon')
        assert success or steps[-1]['action']==0 or len(steps)==32
        counts['episodes']+=1;counts[outcome]+=1;counts['actions']+=len(steps)
        counts['collisions']+=sum(s['collision'] for s in steps);episode_ids.append(e['episode_id'])
        for label in [f'domain/{e["domain"]}',f'distance/{e["distance_group"]}',f'category/{e["category"]}']:
            groups[label]['episodes']+=1;groups[label]['success']+=success;groups[label][outcome+'_outcome']+=1
        seen=set();ret=0.
        for j,s in enumerate(steps):
            a=s['action'];counts['all_repeated_actions']+=a in seen;seen.add(a)
            observed=s.get('observed_ids',s.get('observation_tokens'));state=s.get('state_ids',s.get('state'))
            for f in range(4):factors[f][observed[f]]+=1;memory[f][state[f]]+=1
            counts['sampled_stop']+=a==0
            if j:
                prev=steps[j-1];counts['transitions']+=1;counts['consecutive_repeated_actions']+=a==prev['action']
                if prev['collision']:
                    counts['after_collision']+=1;counts['repeat_after_collision']+=a==prev['action']
                    counts['observation_unchanged_after_collision']+=observed==prev.get('observed_ids',prev.get('observation_tokens'))
                    counts['memory_unchanged_after_collision']+=state==prev.get('state_ids',prev.get('state'))
                    counts['visual_tokens_unchanged_after_collision']+=s['patch_tokens']==prev['patch_tokens']
            if training:
                r=s['reward'];rt=s['reward_terms'];maximum_reward_error=max(maximum_reward_error,abs(sum(rt.values())-r))
                ret+=r
                for k,v in rt.items():terms[k]+=v
        if training:outcomes[outcome].append(ret)
        if not success:failure_distances.append(t['metrics']['distance'])
    n=counts['episodes'];assert n
    return {'counts':dict(counts),'success_rate':counts['success']/n,'unique_episode_ids':len(set(episode_ids)),
     'collision_rate':counts['collisions']/counts['actions'],
     'repeat_after_collision_rate':counts['repeat_after_collision']/max(1,counts['after_collision']),
     'observation_used_ids_per_factor':[len(x) for x in factors],
     'observation_majority_fraction_per_factor':[max(x.values())/sum(x.values()) for x in factors],
     'memory_used_ids_per_factor':[len(x) for x in memory],
     'groups':{k:dict(v) for k,v in groups.items()},
     'mean_reward_terms_per_episode':{k:v/n for k,v in terms.items()},
     'return_by_outcome':{k:{'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'min':min(v),'max':max(v)} for k,v in outcomes.items()},
     'reward_sum_max_error':maximum_reward_error,
     'failed_final_distance_mean':statistics.mean(failure_distances) if failure_distances else None}

result={'time':time.time(),'method':'Offline audit of existing completed trajectories only; no new rollout, no training, no fixed80 inspection.',
 'development':{},'training_windows':{}}
source=B/'observation_inverse_v76/observed_inverse/evaluations/step_000040'
result['development']['supervised_source']=summarize(sorted(source.glob('*/trace.json')))
for folder in sorted((P/'evaluations').iterdir()):
    if (folder/'summary.json').exists():
        r=summarize(sorted(folder.glob('*/trace.json')));assert r['counts']['episodes']==80
        assert r['counts']['success']==read(folder/'summary.json')['successes']
        result['development'][folder.name]=r
for start in [1,11,21,31]:
    paths=[p for i in range(start,start+10) for p in sorted((P/'rollouts'/f'step_{i:06d}').glob('*.json'))]
    assert len(paths)==160
    r=summarize(paths,training=True);assert r['reward_sum_max_error']<1e-5
    result['training_windows'][f'{start}-{start+9}']=r
out=B/'saved_ppo_audit_v83';out.mkdir(exist_ok=True)
(out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
