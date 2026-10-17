"""Small checks for window boundaries and pre-correction prediction accounting."""
import json
from pathlib import Path
from collections import defaultdict
from metrics import prepare, summarize
from remote_stream import compact_episode


def test_state_pairs():
    def step(n, observed, predicted, digest='a', collision=False):
        return dict(step=n, observed_id=observed, action=1, collision=collision,
                    proof=dict(program_sha256=digest, states=[999, predicted], corrected=predicted))
    record = dict(episode=dict(episode_id='fixture', distance_group='near', domain='normal', category='chair'),
                  programs=[dict(start=0), dict(start=3)], steps=[
                      step(0, 1, 7), step(1, 7, 8, collision=True), step(2, 9, 4),
                      step(3, 4, 5), step(4, 5, 6)], success=False, stopped=False,
                  planning_failed=True, **{'return': -12})
    item = compact_episode(record, 1, 0, 0)
    # Pairs 0->1 and 3->4 correct; 1->2 incorrect; 2->3 excluded at renewal.
    assert item['transition_checked'] == 3 and item['transition_correct'] == 2
    assert item['clear_checked'] == 2 and item['clear_correct'] == 2
    assert item['steps'] == 5  # Last step has no post-action frame in the trace.


def test_saved_history():
    path = Path(__file__).with_name('cache.json')
    state = json.loads(path.read_text(encoding='utf-8'))
    attempts = sorted(state['attempts'].values(), key=lambda a: (a['update'], a['rank'], a['index']))
    assert attempts, 'No synchronized attempts'
    by_update = defaultdict(list)
    for attempt in attempts:
        by_update[attempt['update']].append(attempt)
    verified = 0
    for update, batch in by_update.items():
        log = state['train'][str(update)]
        if len(batch) != log['episodes']:
            continue
        assert sum(a['success'] for a in batch) == log['successes']
        assert sum(a['steps'] for a in batch) == log['executed_steps']
        assert sum(a['programs'] for a in batch) == log['rules_generated']
        verified += 1
    for k in (1, 6, 127, 128, 1024, 20000):
        result = prepare(state, k)
        expected = summarize(attempts[-k:])
        assert result['window'] == expected
        for row in result['performance']:
            prefix = [a for a in attempts if a['update'] <= row['update']][-k:]
            check = summarize(prefix)
            for key in ('success_rate', 'transition_accuracy', 'collision_rate'):
                assert row[key] == check[key], (k, row['update'], key)
        for field in ('distance', 'domain', 'category'):
            assert sum(g['attempts'] for name, g in result['groups'].items() if name.startswith(field+':')) == len(attempts[-k:])
    print(json.dumps(dict(verified_updates=verified, attempts=len(attempts), windows=[1, 6, 127, 128, 1024, 20000], passed=True)))


if __name__ == '__main__':
    test_state_pairs()
    test_saved_history()
