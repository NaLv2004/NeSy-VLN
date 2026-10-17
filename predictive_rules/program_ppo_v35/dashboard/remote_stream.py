"""Read-only PPO telemetry. No imports from the trainer and no CUDA allocation."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
RUN = ROOT / 'run_program_ppo_v35'
MAX_ATTEMPTS = 20000


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def emit(value):
    print(json.dumps(value, separators=(',', ':'), allow_nan=False), flush=True)


class Tail:
    def __init__(self, path):
        self.path, self.position = path, 0

    def poll(self):
        result = []
        if not self.path.exists():
            return result
        if self.path.stat().st_size < self.position:
            self.position = 0
        with self.path.open('rb') as handle:
            handle.seek(self.position)
            for line in handle:
                if not line.endswith(b'\n'):
                    break
                self.position += len(line)
                try:
                    result.append(json.loads(line))
                except ValueError:
                    pass
        return result


def compact_episode(record, update, rank, index):
    steps = record['steps']
    renewed = {p['start'] for p in record['programs']}
    checked = correct = moving = moving_correct = clear = clear_correct = 0
    for previous, following in zip(steps, steps[1:]):
        # A new program may redefine the anonymous IDs. Never compare across it.
        if following['step'] != previous['step'] + 1 or following['step'] in renewed:
            continue
        proof, next_proof = previous['proof'], following['proof']
        if proof['program_sha256'] != next_proof['program_sha256']:
            continue
        expected = int(proof['states'][1])
        observed = int(following['observed_id'])  # Before C[predicted, observed].
        hit = int(expected == observed)
        checked += 1
        correct += hit
        if previous['action'] != 0:
            moving += 1
            moving_correct += hit
            if not previous.get('collision', False):
                clear += 1
                clear_correct += hit
    episode = record['episode']
    success = bool(record['success'])
    reason = ('success' if success else 'no_proof' if record['planning_failed'] else
              'false_stop' if record['stopped'] else 'budget' if len(steps) >= 32 else 'other')
    return dict(key=f'{update}:{rank}:{index}', update=update, rank=rank, index=index,
                episode_id=episode['episode_id'], success=success,
                distance=episode['distance_group'], domain=episode['domain'], category=episode['category'],
                reward=record['return'], steps=len(steps), programs=len(record['programs']),
                collisions=sum(bool(s.get('collision')) for s in steps), reason=reason,
                transition_checked=checked, transition_correct=correct,
                moving_checked=moving, moving_correct=moving_correct,
                clear_checked=clear, clear_correct=clear_correct)


def alive(pid):
    try:
        return bool(pid and Path('/proc', str(pid), 'cmdline').exists())
    except OSError:
        return False


def main():
    manifest = read(RUN / 'manifest.json', {})
    launch = read(ROOT / 'program_ppo_v35_launcher.json', {})
    emit(dict(kind='hello', run=RUN.name, manifest=manifest,
              baseline=launch.get('source_development'), remote_path=str(RUN), time=time.time(),
              metric_version=1, max_attempts=MAX_ATTEMPTS))
    training, validation = Tail(RUN / 'train.jsonl'), Tail(RUN / 'validation.jsonl')
    factored_run = ROOT / 'run_factored_world_v36'
    factored_train = Tail(factored_run / 'rank0.jsonl')
    factored_dev = Tail(factored_run / 'development.jsonl')
    train_rows, pending, pending_dev = {}, set(), set()
    gpu, gpu_at = [], 0
    while True:
        began = time.monotonic()
        new_train, new_dev = training.poll(), validation.poll()
        candidates=('run_onpolicy_refresh_v60','run_goal_equivalence_v57','run_definition_rules_v56','run_temporal_symbols_v55','run_hybrid_replay_v54','run_goal_aware_symbols_v52','run_contextual_symbols_v48','run_boundary_balanced_v43','run_grounded_memory_v42','run_stable_onpolicy_v41','run_probabilistic_rules_v40','run_induced_effects_v39','run_discrete_closure_v38','run_learned_persistence_v37','run_factored_world_v36')
        requested_run=next((ROOT/name for name in candidates if (ROOT/name/'supervisor.json').exists()),factored_run)
        if requested_run != factored_run:
            factored_run=requested_run
            factored_train=Tail(factored_run/'rank0.jsonl');factored_dev=Tail(factored_run/'development.jsonl')
        factored_supervisor = read(factored_run / 'supervisor.json', {})
        emit(dict(kind='factored', train=factored_train.poll(), development=factored_dev.poll(),
            status=read(factored_run / 'rank0.status.json', {}), supervisor=factored_supervisor,
            alive=alive(factored_supervisor.get('pid')), latest=read(factored_run / 'latest.COMMITTED.json'),
            collection=read(ROOT/'onpolicy_successors_v60/status.json',{}),
            collection_ranks=[read(ROOT/'onpolicy_successors_v60'/f'status_rank{r}.json',{}) for r in (0,1)],
            handoff=read(ROOT/('diagnostic_onpolicy_refresh_v60' if (ROOT/'diagnostic_onpolicy_refresh_v60/handoff.status.json').exists() else 'diagnostic_goal_equivalence_v57' if factored_run.name.endswith('v57') else 'diagnostic_definition_rules_v56' if factored_run.name.endswith('v56') else 'diagnostic_temporal_symbols_v55' if factored_run.name.endswith('v55') else 'diagnostic_hybrid_replay_v54')/'handoff.status.json',{}),
            remote_path=str(factored_run), time=time.time()))
        for row in new_train:
            train_rows[int(row['update'])] = row
            pending.add(int(row['update']))
        pending_dev.update(int(row['update']) for row in new_dev)
        if train_rows:
            history_count = math.ceil(MAX_ATTEMPTS / max(1, manifest.get('episodes_per_update', 128)))
            eligible = sorted(train_rows)[-history_count:]
            pending.intersection_update(eligible)
        now = time.time()
        if now - gpu_at >= 30:
            try:
                text = subprocess.check_output(['nvidia-smi', '-i', '6,7',
                    '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=8)
                gpu = [dict(zip(['index', 'used_mib', 'total_mib', 'utilization', 'temperature'],
                                map(int, line.split(',')))) for line in text.splitlines()]
                gpu_at = now
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        supervisor = read(RUN / 'supervisor.json', {})
        ranks = []
        for rank in range(2):
            status = read(RUN / f'rank{rank}/status.json', {})
            error = read(RUN / f'rank{rank}/error.json')
            if error and error.get('time', 0) < supervisor.get('started', 0):
                error = None
            ranks.append(dict(rank=rank, status=status, error=error, alive=alive(status.get('pid'))))
        telemetry = dict(time=now, gpu=gpu, gpu_time=gpu_at, supervisor=supervisor,
                         supervisor_alive=alive(supervisor.get('pid')), ranks=ranks,
                         status=read(RUN / 'status.json'), latest=read(RUN / 'latest.COMMITTED.json'),
                         best=read(RUN / 'best.COMMITTED.json'))
        emit(dict(kind='telemetry', telemetry=telemetry, train=new_train, development=new_dev,
                  pending_history_updates=len(pending)))
        # Backfill bounded batches of saved traces; no simulator or model is run.
        for update in sorted(pending)[:4]:
            records = []
            log = train_rows[update]
            for rank in range(2):
                directory = RUN / f'rank{rank}/rollouts/update_{update:06d}'
                candidates = {}
                for path in directory.glob('*/trace.json'):
                    index = int(path.parent.name.split('_', 1)[0])
                    modified = path.stat().st_mtime
                    # Failed startup attempts can leave different task folders at
                    # the same batch index. Match the committed update's interval.
                    if not log['time'] - log['seconds'] - 2 <= modified <= log['time'] + 2:
                        continue
                    if index not in candidates or modified > candidates[index][0]:
                        candidates[index] = (modified, path)
                for index, (_, path) in sorted(candidates.items()):
                    record = read(path)
                    if record is None:
                        continue
                    records.append(compact_episode(record, update, rank, index))
            expected = int(log['episodes'])
            audited = (len(records) == expected and sum(r['success'] for r in records) == log['successes']
                       and sum(r['steps'] for r in records) == log['executed_steps']
                       and sum(r['programs'] for r in records) == log['rules_generated'])
            if audited:
                emit(dict(kind='attempts', update=update, expected=expected, found=len(records),
                          audited_against_training_log=True, attempts=records))
                pending.discard(update)
            else:
                emit(dict(kind='trace_audit_pending', update=update, found=len(records), expected=expected))
        for update in sorted(pending_dev)[:2]:
            totals = dict(transition_checked=0, transition_correct=0, clear_checked=0,
                          clear_correct=0, steps=0, episodes=0)
            for rank in range(2):
                directory = RUN / f'rank{rank}/development/update_{update:06d}'
                for index, path in enumerate(sorted(directory.glob('*/trace.json'))):
                    record = read(path)
                    if record is None:
                        continue
                    item = compact_episode(record, update, rank, index)
                    totals['episodes'] += 1
                    for key in totals:
                        if key != 'episodes':
                            totals[key] += item[key]
            if totals['episodes'] == 80:
                emit(dict(kind='development_transitions', update=update, transitions=totals))
                pending_dev.discard(update)
        time.sleep(max(.1, 5 - (time.monotonic() - began)))


if __name__ == '__main__':
    main()
