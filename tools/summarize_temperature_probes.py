import json,statistics,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010')
records=json.loads((base/'paired_goal_probe_v12/input_snapshot/pairs.json').read_text())
records=[r for r in records if r['both_successful'] and r['same_initial_rgb'] and r['first_actions_differ']]
heldout=set(sorted({r['scene'] for r in records})[::4])
groups={name:[r for r in records if (r['scene'] in heldout)==is_holdout] for name,is_holdout in [('train',False),('heldout',True)]}
outputs={}
for name,path in [('T1',base/'branch_rule_capacity_v16/branch_current/result.json'),
                  ('T2',base/'branch_temperature_v18/temperature_2/result.json'),
                  ('T4',base/'branch_temperature_v18/temperature_4/result.json')]:
    result=json.loads(path.read_text());assert result['state']=='complete'
    last=result['learning'][-1];out={'step':last['step'],'seconds':last['seconds'],'groups':{}}
    norms=[row['unclipped_gradient_norm'] for row in result['learning'] if 'unclipped_gradient_norm' in row]
    if norms:out['logged_gradient_norm']={'median':statistics.median(norms),'max':max(norms),'last':norms[-1]}
    for split,rows in groups.items():
        measure=last[split];labels=measure['labels'];predictions=measure['predictions']
        flat=[b for r in rows for b in r['branches']]
        assert labels==[b['frames'][0]['action'] for b in flat]
        buckets={}
        for label,mask in [('visible',[b['initial_metrics']['visible_pixels']>=16 for b in flat]),
                           ('invisible',[b['initial_metrics']['visible_pixels']==0 for b in flat]),
                           ('tiny_visibility',[0<b['initial_metrics']['visible_pixels']<16 for b in flat])]:
            selected=[i for i,v in enumerate(mask) if v];correct=sum(predictions[i]==labels[i] for i in selected)
            buckets[label]={'n':len(selected),'correct':correct,'accuracy':correct/len(selected) if selected else None}
        out['groups'][split]={'accuracy':measure['accuracy'],'nll':measure['nll'],
            'different_goal_changes_action':measure['different_goal_changes_action'],
            'goal_swap_accuracy':measure['goal_swap_accuracy'],'image_swap_accuracy':measure['image_swap_accuracy'],
            'visibility_strata':buckets}
    outputs[name]=out
summary={'time':time.time(),'results':outputs,'fixed_validation_used':False,
    'conclusion':'Temperature reduced confidence but does not establish improved held-out action prediction or closed-loop navigation.',
    'interpretation_limit':'Oracle first-action labels for invisible goals can depend on privileged goal locations. Exact agreement is a capacity diagnostic, not navigation success.'}
(base/'branch_temperature_v18/completion_inspection.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
