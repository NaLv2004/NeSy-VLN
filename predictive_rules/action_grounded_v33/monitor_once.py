"""Inspect the existing formal run, at most once per 600 remote seconds.

This does not launch evaluations, rollouts, training, or recovery processes.
"""
import json
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT = ROOT / 'run_action_grounded_v33'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def rows(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrent append may not have completed yet.
    return result


previous = ROOT / 'action_grounded_v33_monitor_latest.json'
startup = read(ROOT / 'action_grounded_v33_startup.json')
last = read(previous) or startup
now = time.time()
minimum = last['observed_at'] + 600
if now < minimum:
    print(json.dumps(dict(deferred=True, next_check_not_before=minimum, current_time=now)))
    raise SystemExit

launch = read(ROOT / 'action_grounded_v33_launcher.json')
collector = read(ROOT / 'raw_successor_features_v31/parallel_render_launcher.json')
processes = {}
for item in (launch, collector):
    pid = item['pid']
    path = Path('/proc', str(pid), 'cmdline')
    processes[pid] = path.read_bytes().replace(b'\0', b' ').decode() if path.exists() else None

training = rows(OUT / 'train.jsonl')
metrics = [
    'transition_accuracy', 'counterfactual_accuracy', 'proof_action_accuracy',
    'subgoal_exact_accuracy', 'online_target_state_agreement',
    'hard_rule_planning_action_accuracy', 'hard_rule_planning_loss',
    'world_gradient_norm', 'definitions_gradient_norm', 'seconds',
    'branch_rule_inverse_accuracy', 'branch_rule_inverse_loss',
]
windows = {}
for label, sample in [('first100', training[:100]), ('last100', training[-100:])]:
    if sample:
        windows[label] = dict(
            first_update=sample[0]['update'], last_update=sample[-1]['update'],
            count=len(sample),
            metrics={key: statistics.mean(v[key] for v in sample if key in v)
                     for key in metrics if any(key in v for v in sample)},
        )

collection = ROOT / 'raw_successor_features_v31'
result = dict(
    observed_at=now, processes=processes, status=read(OUT / 'status.json'),
    latest_commit=read(OUT / 'latest.COMMITTED.json'), training_windows=windows,
    validation=rows(OUT / 'validation.jsonl'),
    evaluation_error=read(OUT / 'evaluation_error.json'),
    collector_status=read(collection / 'parallel_render/status.json'),
    raw_positions=sum(read(f)['positions'] for f in collection.glob('counterfactual_*/*.json')),
    gpu=subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader'], text=True),
    log_tail=(ROOT / 'action_grounded_v33.log').read_text(errors='replace')[-1500:],
    collector_log_tail=(collection / 'parallel_render.log').read_text(errors='replace')[-1000:],
)
previous.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
