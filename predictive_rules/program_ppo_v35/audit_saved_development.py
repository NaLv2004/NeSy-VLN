"""Compare already-saved formal evaluations; no simulator or model inference.

At the initial frame, every frozen definition/perception input should match.
This separates changed program choices from floating-point grounding changes.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')


def read(path):return json.loads(path.read_text())


def first(record):
    if record['steps']:return record['steps'][0]
    failed=record.get('failed_proof',{})
    return dict(observed_id=failed.get('observed'),proof=failed,action=None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--update',type=int,required=True);args=parser.parse_args()
    launch=read(ROOT/'program_ppo_v35_launcher.json')
    baseline=Path(launch['source_checkpoint']).parent/f'development_step_{launch["source_development"]["update"]:06d}'
    run=ROOT/'run_program_ppo_v35'
    rows=[];failures=Counter();initial_ids=Counter();goals=Counter()
    for rank in range(2):
        folder=run/f'rank{rank}/development/update_{args.update:06d}'
        if not (folder/'summary.json').exists():raise RuntimeError('Scheduled formal development is not complete')
        for path in folder.glob('*/trace.json'):
            current=read(path);eid=current['episode']['episode_id'];old=read(baseline/eid/'trace.json')
            assert current['episode']==old['episode'],'Different task/start pose; comparison invalid'
            a=first(old);b=first(current)
            initial_ids[b['observed_id']]+=1
            goal=current['programs'][0]['goal'];goals[goal]+=1
            failed='success' if current['success'] else 'no_proof' if current['planning_failed'] else 'false_stop' if current['stopped'] else 'budget'
            failures[failed]+=1
            rows.append(dict(episode_id=eid,old_success=bool(old.get('metrics',{}).get('success',False)) and not old['planning_failed'],
                new_success=current['success'],failure=failed,
                initial_id_old=a['observed_id'],initial_id_new=b['observed_id'],
                initial_id_unchanged=a['observed_id']==b['observed_id'],
                initial_goal_old=old['programs'][0]['goal'],initial_goal_new=goal,
                first_action_old=a['action'],first_action_new=b['action'],
                steps=len(current['steps']),programs=len(current['programs'])))
    assert len(rows)==80 and len({r['episode_id'] for r in rows})==80
    result=dict(update=args.update,episodes=80,read_only_existing_evidence=True,
        first_observation_id_unchanged=sum(r['initial_id_unchanged'] for r in rows),
        first_goal_unchanged=sum(r['initial_goal_old']==r['initial_goal_new'] for r in rows),
        first_action_unchanged=sum(r['first_action_old']==r['first_action_new'] for r in rows),
        gained_successes=[r['episode_id'] for r in rows if r['new_success'] and not r['old_success']],
        lost_successes=[r['episode_id'] for r in rows if r['old_success'] and not r['new_success']],
        failures=dict(failures),initial_ids=dict(initial_ids),initial_goals=dict(goals),rows=rows)
    path=run/f'saved_development_audit_{args.update:06d}.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
