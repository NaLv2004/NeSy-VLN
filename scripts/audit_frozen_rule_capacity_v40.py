"""Offline upper bounds on frozen symbolic programs, with no image inputs.

The LP freely mixes generated conclusions, so its reachability is an upper
bound on what any rule weighting could do, not a executable policy or score.
"""
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

root = Path(__file__).resolve().parents[1]
source = root / 'output/frozen_rule_learning_v39/symbols.json'
data = json.loads(source.read_text())
out = {'time': time.time(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
       'uses_images': False, 'uses_original_development_or_fixed80': False,
       'qualification': 'Static training-pool audit; expert action agreement is not navigation success. '
       'The LP allows arbitrary rule mixing and therefore only provides an upper bound.', 'categories': {}}

def state_groups(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row['codes'])].append(row)
    return groups

def metrics(rows, heads, program, sl):
    codes = np.array([r['codes'] for r in rows])
    literals = np.concatenate([np.ones((len(rows), 1)), codes, 1-codes], 1)
    mandatory = literals @ np.array(program['premises'])[sl, 0].T
    conditions = np.array(program['conjunctive_conditions'])[sl]
    matched = conditions[None, :, :, 0] + conditions[None, :, :, 1] * codes[:, None] + conditions[None, :, :, 2] * (1-codes[:, None])
    logs = np.log(mandatory.clip(1e-30)) + np.log(matched.clip(1e-30)).sum(-1) + np.log(np.array(program['weights'])[sl])[None]
    mass = np.exp(logs-logs.max(-1, keepdims=True)); mass /= mass.sum(-1, keepdims=True)
    choices = (mass @ heads).argmax(-1)
    groups = state_groups(rows)
    majority = sum(max(collections.Counter(r['label'] for r in group).values()) for group in groups.values())
    return {'n': len(rows), 'distinct_states': len(groups), 'constant_bits': np.where(codes.max(0)==codes.min(0))[0].tolist(),
            'greedy_correct': int(sum(int(a)==r['label'] for a, r in zip(choices, rows))),
            'greedy_counts': dict(collections.Counter(map(int, choices))),
            'empirical_deterministic_action_agreement_ceiling': majority/len(rows),
            'conflicting_state_groups': sum(len({r['label'] for r in g}) > 1 for g in groups.values()),
            'ambiguous_examples': sum(len(g) for g in groups.values() if len({r['label'] for r in g}) > 1),
            'mean_rule_entropy': float(-(mass*np.log(mass.clip(1e-30))).sum(-1).mean())}

for category, program in data['programs'].items():
    rows = [r for r in data['examples'] if r['category']==category]
    train = [r for r in rows if not r['heldout']]; holdout = [r for r in rows if r['heldout']]
    n = len(program['weights'])//6; sl = slice(2*n, 4*n)
    heads = np.array(program['action_heads'])[sl]
    assert np.allclose(heads.sum(1), 1, atol=1e-5)
    actions = []
    for action in range(heads.shape[1]):
        others = [a for a in range(heads.shape[1]) if a != action]
        # Variables are normalized nonnegative rule masses and winning margin.
        inequalities = np.concatenate([(heads[:, others]-heads[:, action, None]).T,
                                       np.ones((len(others), 1))], 1)
        lp = linprog(np.r_[np.zeros(len(heads)), -1.], A_ub=inequalities, b_ub=np.zeros(len(others)),
                     A_eq=np.array([np.r_[np.ones(len(heads)), 0.]]), b_eq=[1.],
                     bounds=[(0, 1)]*len(heads)+[(None, None)], method='highs')
        assert lp.success, lp.message
        probs = lp.x[:-1] @ heads
        margin = float(probs[action]-np.max(probs[others]))
        assert abs(margin-lp.x[-1]) < 1e-6
        actions.append({'action': action, 'max_winning_margin': margin,
                        'strictly_reachable_under_arbitrary_mixing': margin > 1e-7,
                        'rule_argmax_count': int((heads.argmax(-1)==action).sum()),
                        'max_single_rule_probability': float(heads[:, action].max()),
                        'train_label_count': sum(r['label']==action for r in train),
                        'holdout_label_count': sum(r['label']==action for r in holdout),
                        'witness_rules': [{'rule': int(i+2*n), 'mass': float(w)} for i,w in enumerate(lp.x[:-1]) if w>1e-6]})
    impossible = {a['action'] for a in actions if a['max_winning_margin'] < -1e-7}
    groups = state_groups(train)
    seen = [r for r in holdout if tuple(r['codes']) in groups]
    train_majority = {k: collections.Counter(r['label'] for r in g).most_common(1)[0][0] for k,g in groups.items()}
    out['categories'][category] = {'train': metrics(train, heads, program, sl),
        'holdout': metrics(holdout, heads, program, sl), 'conclusion_capacity': actions,
        'provably_impossible_greedy_actions_even_with_arbitrary_rule_mixing': sorted(impossible),
        'impossible_expert_labels': {name: sum(r['label'] in impossible for r in subset)
                                    for name,subset in [('train',train),('holdout',holdout)]},
        'holdout_identical_to_training_state_count': len(seen),
        'holdout_identical_state_train_majority_correct': sum(train_majority[tuple(r['codes'])]==r['label'] for r in seen)}

out['replay_check'] = {name: sum(c[name]['greedy_correct'] for c in out['categories'].values()) for name in ['train','holdout']}
assert out['replay_check'] == {'train': 235, 'holdout': 53}, out['replay_check']
target = root/'output/frozen_rule_capacity_v40.json'
target.write_text(json.dumps(out, indent=2))
print(json.dumps({c:{k:v for k,v in x.items() if k!='conclusion_capacity'} for c,x in out['categories'].items()}, indent=2))
