"""Count-weighted dashboard statistics from saved attempts, never averaged rates."""
from collections import defaultdict, deque
import math
import statistics


def ratio(a, b):
    return a / b if b else None


FIELDS = ('steps', 'programs', 'collisions', 'transition_checked', 'transition_correct',
          'moving_checked', 'moving_correct', 'clear_checked', 'clear_correct')


def finish(totals):
    values = {key: totals.get(key, 0) for key in (*FIELDS, 'attempts', 'successes')}
    values.update(success_rate=ratio(values['successes'], values['attempts']),
                  collision_rate=ratio(values['collisions'], values['steps']),
                  transition_accuracy=ratio(values['transition_correct'], values['transition_checked']),
                  transition_coverage=ratio(values['transition_checked'], values['steps']),
                  clear_accuracy=ratio(values['clear_correct'], values['clear_checked']),
                  steps_per_program=ratio(values['steps'], values['programs']),
                  mean_return=ratio(totals.get('reward', 0), values['attempts']))
    return values


def accumulate(totals, attempt, sign=1):
    for key in (*FIELDS, 'reward'):
        totals[key] += sign * attempt.get(key, 0)
    totals['attempts'] += sign
    totals['successes'] += sign * attempt['success']


def summarize(attempts):
    totals = defaultdict(float)
    for attempt in attempts:
        accumulate(totals, attempt)
    return finish(totals)


def prepare(state, k):
    rows = sorted(state['train'].values(), key=lambda r: r['update'])
    attempts = sorted(state['attempts'].values(), key=lambda a: (a['update'], a['rank'], a['index']))
    window = attempts[-k:]
    groups = defaultdict(list)
    reasons = defaultdict(int)
    for attempt in window:
        for field in ('distance', 'domain', 'category'):
            groups[field + ':' + attempt[field]].append(attempt)
        reasons[attempt['reason']] += 1
    # A true K-attempt window at every completed update endpoint.
    by_update = defaultdict(list)
    for attempt in attempts:
        by_update[attempt['update']].append(attempt)
    queue = deque()
    running = defaultdict(float)
    distance_running = defaultdict(lambda: defaultdict(float))
    performance = []
    for update, batch in sorted(by_update.items()):
        for attempt in batch:
            queue.append(attempt)
            accumulate(running, attempt)
            accumulate(distance_running[attempt['distance']], attempt)
            if len(queue) > k:
                removed = queue.popleft()
                accumulate(running, removed, -1)
                accumulate(distance_running[removed['distance']], removed, -1)
        summary = finish(running)
        summary['difficulty'] = {key: finish(value) for key, value in distance_running.items()}
        summary['update'] = update
        summary['complete_update'] = len(batch) == state['train'].get(str(update), {}).get('episodes')
        performance.append(summary)
    recent = rows[-10:]
    # Bound browser drawing cost, retaining both ends of history.
    stride = max(1, math.ceil(len(rows) / 2000))
    chart_rows = rows[::stride]
    if rows and (not chart_rows or chart_rows[-1]['update'] != rows[-1]['update']):
        chart_rows.append(rows[-1])
    development = []
    for row in sorted(state['development'].values(), key=lambda r: r['update']):
        row = dict(row)
        transition = state.get('development_transitions', {}).get(str(row['update']))
        if transition:
            row['transition_accuracy'] = ratio(transition['transition_correct'], transition['transition_checked'])
            row['transition_counts'] = transition
        development.append(row)
    return dict(k=k, window=summarize(window), groups={key: summarize(value) for key, value in groups.items()},
                reasons=dict(reasons), performance=performance,
                train=chart_rows, development=development,
                latest_train=rows[-1] if rows else None,
                total_updates=len(rows), total_attempts=sum(r['episodes'] for r in rows),
                loaded_attempts=len(attempts), last_attempt_update=attempts[-1]['update'] if attempts else None,
                recent_mean_seconds=statistics.mean(r['seconds'] for r in recent) if recent else None,
                recent_trajectories_per_minute=60 * sum(r['episodes'] for r in recent) / sum(r['seconds'] for r in recent) if recent else None,
                recent_attempts=attempts[-20:][::-1], trace_coverage=state.get('trace_coverage', {}))
