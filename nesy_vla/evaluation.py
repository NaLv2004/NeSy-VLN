from collections import defaultdict
from .policy import rollout,rollout_many
from .reporting import save_trace


def evaluate_tasks(model,sim,tasks,folder,stochastic=False,revision=True):
    rows=[]
    simulators=sim if isinstance(sim,list) else [sim]
    for offset in range(0,len(tasks),len(simulators)):
        batch=tasks[offset:offset+len(simulators)]
        policies,metrics=rollout_many(model,simulators[:len(batch)],batch,stochastic,revision)
        for episode,policy,result in zip(batch,policies,metrics):
            save_trace(folder/episode['episode_id'],policy,episode,result)
            steps=[r for c in policy.chunks for r in c['rows']]
            rows.append({'episode_id':episode['episode_id'],'scene':episode['scene'],
                'domain':episode['domain'],'category':episode['category'],'distance_group':episode['distance_group'],
                'success':bool(result.get('success')),'steps':len(steps),
                'collisions':sum(r['collision'] for r in steps),'revision_count':max(0,len(policy.chunks)-1),
                'repeated_action_pairs':sum(a['action']==b['action'] for a,b in zip(steps,steps[1:])),
                'prediction_error_mean':sum(r['prediction_error'] for r in steps)/max(1,len(steps)),
                'sampling':'stochastic' if stochastic else 'greedy','online_revision':revision})
    return rows


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:
        for key in ['domain','category','distance_group']:groups[f'{key}/{r[key]}'].append(r)
    def metric(items):
        steps=sum(r['steps'] for r in items)
        return {'episodes':len(items),'successes':sum(r['success'] for r in items),
            'success_rate':sum(r['success'] for r in items)/max(1,len(items)),
            'collision_action_rate':sum(r['collisions'] for r in items)/max(1,steps),
            'mean_steps':steps/max(1,len(items))}
    return dict(metric(rows),groups={k:metric(v) for k,v in groups.items()},max_actions=32)
