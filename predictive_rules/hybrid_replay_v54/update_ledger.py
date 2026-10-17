"""Reconcile the goal ledger from an actual remote process observation."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

record=read(HERE/'experiment_record.json')
snapshot=read(HERE/'last_observation.json')
progress=read(HERE.parent/'goal_progress.json')
live=[p['pid'] for p in snapshot['processes'] if p['command']]
record.update(previous_goal_turn_classification='progress: restarted requested dashboard and verified fresh remote telemetry',
    current_goal_turn_classification='progress: audited saved full-dev navigation and matched all-action training outcomes; inspected current hybrid replay training with exact process handles',
    last_observation=snapshot,last_remote_check_unix=snapshot['time'],
    next_regular_check_not_before_unix=snapshot['time']+630,
    live_process_ids=live,phase='training' if snapshot['supervisor']['state']=='running' else snapshot['supervisor']['state'])
record['verified_collection']=snapshot['collection']
record['verified_handoff']=snapshot['handoff']
record['verified_training']=dict(supervisor=snapshot['supervisor'],rank_states=snapshot['ranks'],
    latest=snapshot['latest'],best=snapshot['best'],gpu_status=snapshot['gpus'],free_bytes=snapshot['free_bytes'])
if snapshot['development']:
    record['completion_audit']['new_v54_validation']=[dict(update=r['update'],successes=r['successes'],episodes=r['episodes']) for r in snapshot['development']]
for filename in ('navigation_baseline.json','navigation_comparison.json','action_selection_bias.json','retention.json'):
    path=HERE/filename
    if path.exists():record.setdefault('current_turn_diagnostics',{})[filename]=read(path)
record['diagnostic_conclusions']=[
    'Pinned source on120 train-only visited trajectories: chosen moving outcome23.10% accurate, all16 moving outcomes22.06%; broad model weakness rather than uniquely bad action selection.',
    'Old full-dev80 best2000 has11 successes, next observation19.45%, changed observation12.84%;106 of192 repeatedly visited within-program state/action rows produced conflicting next observation IDs.',
    'These are offline diagnostics, no added simulator rollouts, no validation gradients, no neural planning replacement. New v54 success improvement requires full scheduled navigation evaluation.']
if (HERE/'navigation_comparison.json').exists():
    comparison=read(HERE/'navigation_comparison.json')['runs']
    record['diagnostic_conclusions'].append('v54 step500 full80:9 successes versus source11; next-O26.48% versus19.45%; changed-O18.37% versus12.84%. Better factual prediction has not yet produced better navigation. Colliding transition rate48.74% versus40.08%, trajectories differ.')
record['research_considered_this_turn']=[
    dict(url='https://arxiv.org/abs/2106.04379',finding='Inverse dynamics plus temporal contrastive supervision can support Markov abstractions; current objective contains inverse but no temporal contrastive term.',status='Reviewed; not implemented or claimed proven for this POMDP.'),
    dict(url='https://proceedings.mlr.press/v202/tang23d.html',finding='Self-predictive representation learning depends on predictor/encoder time scales and stop-gradient targets.',status='Reviewed; existing code already uses a slower encoder and detached target labels; do not relabel that as a new change.')]
record['next_action']='Keep exact live v54 training until its scheduled full80 results are available. Compare physical success, predicted-M/current-O accuracy, code drift and action-selection bias. Do not run finalfixed80 or repeat completed feature probes. Regular process checks at least600s apart.'
progress.update(phase=record['phase'],last_remote_check_unix=snapshot['time'],
    next_regular_check_not_before_unix=snapshot['time']+630,
    supervisor_pid=snapshot['supervisor']['pid'],rank_pids=[r['pid'] for r in snapshot['ranks']],
    latest_actual_update=min(r['update'] for r in snapshot['ranks']),collection_state=snapshot['collection'],
    next_action=record['next_action'])
record['clock_note']='Use remote timestamps for check gates; reserve30s against the observed approximately21s client/server clock offset.'
if (HERE/'retention.json').exists():
    maintenance=read(HERE/'retention.json')
    record['storage_maintenance']=dict(time=maintenance['time'],removed_count=maintenance['removed_count'],
        reclaimed_bytes=maintenance['estimated_bytes'],free_after=maintenance['free_after'],audit=str(HERE/'retention.json'))
write(HERE/'experiment_record.json',record);write(HERE.parent/'goal_progress.json',progress)
print(json.dumps(dict(phase=progress['phase'],update=progress['latest_actual_update'],next_check=progress['next_regular_check_not_before_unix'])))
