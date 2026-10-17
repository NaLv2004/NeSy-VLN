"""Replay training-only actions and actual immediate STOP from identical starts.

Alternative rewards are evaluated on the same observed transitions, not supplied
to a policy. No evaluation tasks, invented predicates, or image-action module.
"""
import collections,copy,hashlib,json,os,statistics,sys,time
from pathlib import Path

BASE=Path('/home/user/xjy/nesy_vla_20261010')
REF=Path('/home/user/xjy/strips_wm_navigation_20261009')
OUT=BASE/'reward_diagnostic_v11';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(REF))
from stripsnav.sim_adapter import NavigationEnvironment

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
cfg=read(REF/'configs/experiment.json');cfg.update(training_horizon=32,evaluation_horizon=32)
lock=read(REF/'research/split_lock.json');data=Path(cfg['source_data'])
assert sha(data/'train.json')==lock['hashes']['train.json']
assert sha(data/'fixed80.json')==lock['hashes']['fixed80.json']
train=set(lock['train_scenes'])-set(lock['development_scenes'])
by_id={t['episode_id']:t for t in read(data/'train.json') if t['scene'] in train}
demo=collections.defaultdict(list)
for row in read(BASE/'data/lossless_demonstrations_v7/examples.json'):
    assert row['episode_id'] in by_id
    demo[row['episode_id']].append(row)
selected=[]
for domain in ['normal','dense']:
    for category in ['chair','table']:
        choices=[(eid,sorted(rows,key=lambda r:r['image'])) for eid,rows in demo.items()
                 if by_id[eid]['domain']==domain and by_id[eid]['category']==category]
        for eid,rows in sorted(choices,key=lambda item:(-len(item[1]),item[0]))[:3]:
            selected.append({'id':eid,'source':'long_successful_demonstration','actions':[r['action'] for r in rows]})
traces=BASE/'symbolic_joint_v9/runs/joint/traces'
failures=[]
for path in sorted(traces.glob('update_*/*/trajectory.json')):
    try:t=read(path)
    except (OSError,json.JSONDecodeError):continue
    if not t['metrics'].get('success') and len(t['steps'])==32 and t['steps'][-1]['action']!=0:
        assert t['episode_id'] in by_id
        failures.append({'id':t['episode_id'],'source':'sampled_horizon_failure',
                         'trace':str(path),'actions':[r['action'] for r in t['steps']]})
seen=set()
for item in reversed(failures):
    if item['id'] in seen:continue
    seen.add(item['id']);selected.append(item)
    if len(seen)==12:break
alternatives={
    'original':{},
    'false_stop_40':{'false_stop':-40.},
    'false_stop_100':{'false_stop':-100.},
    'bounded_repeat':{'repeat_action':-.10,'repeat_action_growth':-.025,'repeat_action_max':-.25},
    'bounded_repeat_stop40':{'repeat_action':-.10,'repeat_action_growth':-.025,'repeat_action_max':-.25,'false_stop':-40.},
}
env=NavigationEnvironment(cfg,gpu=0,evaluation=False);results=[]
try:
    for item in selected:
        ep=by_id[item['id']];env.reset(ep);initial=env.private_metrics()
        if initial['eligible']:continue
        immediate=env.step(0)
        assert not immediate['metrics']['success'] and immediate['done']
        env.reset(ep);states={}
        for name,overrides in alternatives.items():
            reward=copy.deepcopy(env.env.reward);reward.config=dict(reward.config,**overrides)
            states[name]=reward
        rewards={name:[] for name in alternatives};terms={name:collections.Counter() for name in alternatives}
        samples=[]
        for action in item['actions']:
            response=env.step(action);m=response['metrics'];f=response['feedback']
            for name,reward in states.items():
                scored=reward.update(m['distance'],m['visible_pixels'],response['rgb'],action==0,
                    f['collision'],f['invalid'],f['travelled_m'],json.dumps({'opaque_action':int(action)}))
                if name=='original':assert abs(scored['reward']-response['reward'])<1e-7
                rewards[name].append(scored['reward'])
                for k,v in scored['terms'].items():terms[name][k]+=v
            samples.append({'action':action,'distance':m['distance'],'visible_pixels':m['visible_pixels'],
                'collision':f['collision'],'travelled_m':f['travelled_m'],'repeat':m['action_repetition']})
            if response['done']:break
        comparison={}
        for name,rs in rewards.items():
            stop_reward=immediate['reward']+alternatives[name].get('false_stop',-12.)+12.
            discounted=sum(.99**i*r for i,r in enumerate(rs))
            comparison[name]={'return':sum(rs),'discounted_return':discounted,
                'actual_immediate_stop_with_same_reward':stop_reward,'continuation_minus_stop':discounted-stop_reward,
                'terms':dict(terms[name])}
        record={**item,'scene':ep['scene'],'category':ep['category'],'domain':ep['domain'],
            'initial':initial,'steps':len(samples),'replayed_success':response['metrics']['success'],
            'final_distance':response['metrics']['distance'],'comparisons':comparison,'transitions':samples}
        results.append(record)
        (OUT/'partial.json').write_text(json.dumps(results))
        print(json.dumps({'episode':item['id'],'source':item['source'],'success':record['replayed_success'],
            'steps':len(samples),'original_margin':comparison['original']['continuation_minus_stop']}),flush=True)
finally:env.close()
groups={}
for source in {r['source'] for r in results}:
    rows=[r for r in results if r['source']==source]
    groups[source]={'episodes':len(rows),'actual_successes':sum(r['replayed_success'] for r in rows),
        'mean_steps':statistics.mean(r['steps'] for r in rows),'alternatives':{}}
    for name in alternatives:
        values=[r['comparisons'][name] for r in rows]
        groups[source]['alternatives'][name]={
            'prefer_immediate_stop':sum(v['continuation_minus_stop']<0 for v in values),
            'mean_return':statistics.mean(v['return'] for v in values),
            'mean_continuation_margin':statistics.mean(v['continuation_minus_stop'] for v in values)}
report={'time':time.time(),'training_only':True,'locked_hashes':lock['hashes'],
    'same_start_immediate_stop_executed':True,'all_alternative_rewards_exact_same_transitions':True,
    'original_reward_reproduced_per_step':True,'all_repeat_actions_still_penalized':True,
    'navigation_policy_improvement_tested':False,'alternatives':alternatives,'groups':groups,'episodes':results}
(OUT/'counterfactual.json').write_text(json.dumps(report,indent=2));print(json.dumps({'groups':groups},indent=2))
