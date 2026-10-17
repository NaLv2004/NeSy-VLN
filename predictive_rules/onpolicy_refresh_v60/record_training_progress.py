"""Reconcile saved authoritative observations without probing running training."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

obs = read(HERE / 'training_observation.json')
record = read(HERE / 'experiment_record.json')
sv = obs['supervisor']
live = {p['pid']: p['command'] for p in obs['processes'] if p['command']}
ranks = obs['ranks']
development = obs['development']
record.update(phase=sv['state'], observation=obs, main_training_started=True,
    remote_run=record['remote_future_run'],
    last_remote_check_unix=obs['time'], next_regular_check_not_before_unix=obs['time']+630,
    latest_actual_update=min(r['update'] for r in ranks),
    next_action='Inspect the original training supervisor and ranks after the next630-second gate. Preserve the existing full80 evaluations. The v61 coordinator waits for this exact supervisor to exit before checking the robust exported-table contract and running its full80 evaluation.',
    current_goal_turn_classification='progress', goal_status='active', completion_proven=False)
record['supervisor_live_at_observation'] = sv['pid'] in live
record['rank_pids_live_at_observation'] = [r['pid'] for r in ranks if r['pid'] in live]
record['collection_audit'] = read(HERE/'collection_audit.json')
record['contract'] = read(HERE/'contract.json')
if (HERE/'completed_comparison.json').exists():
    record['completed_comparison'] = read(HERE/'completed_comparison.json')
write(HERE/'experiment_record.json', record)
goal = read(BASE/'goal_progress.json')
goal.update(authoritative_record=str(HERE/'experiment_record.json'), phase=sv['state'],
    remote_run=record['remote_run'], remote_source=record['remote_training_source'],
    main_training_started=True, supervisor_pid=sv['pid'], rank_pids=[r['pid'] for r in ranks],
    latest_actual_update=record['latest_actual_update'], last_remote_check_unix=obs['time'],
    next_regular_check_not_before_unix=obs['time']+630,
    next_action=record['next_action'],
    rank_pid_coverage='Actual /proc command lines captured in training_observation.json; terminal state is checked independently of stale rank files.',
    previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
    independent_experiment=str(BASE/'ensemble_rules_v61/experiment_record.json'),
    goal_status='active', completion_proven=False)
if development:
    row = development[-1]
    goal['latest_verified_full80'] = {k: row[k] for k in ('update','successes','episodes')}
    goal['latest_verified_full80']['run'] = 'onpolicy_refresh_v60'
    for row in development:
        if row['episodes']==80 and row['successes']>goal['best_verified_full80']['successes']:
            goal['best_verified_full80'] = dict(run='onpolicy_refresh_v60', **{k:row[k] for k in ('update','successes','episodes')})
if sv['state']=='finished':
    goal['last_completed_training_update'] = record['latest_actual_update']
write(BASE/'goal_progress.json', goal)
print(json.dumps(dict(phase=record['phase'], update=record['latest_actual_update'],
    next_check=goal['next_regular_check_not_before_unix'], completed=False)))
