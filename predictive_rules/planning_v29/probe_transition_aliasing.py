"""Measure deterministic rule identifiability on recorded training branches.

No optimizer updates, simulator rollout, development example, or semantic
label is used. Each ambiguity group belongs to one fixed generated program.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from planning_model import PlanningObservedWorld, Config
from joint_model import branch_states
from train import ROOT
from predictive_nesy.subgoal_data import TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    used = int(subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True).strip())
    assert used < 256, f'GPU7 is occupied ({used} MiB); do not disturb it'
    torch.set_num_threads(2)
    rng = random.Random(3017)
    saved = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    model = PlanningObservedWorld(Config(**saved['model_config'])).cuda().eval()
    model.load_state_dict(saved['model'])
    features = ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',
                                       ROOT/'replayed_boundary_features_v20', 'cuda')
    totals = Counter()
    details = []
    for record in saved['manifest']['transition_banks']:
        bank = TransitionBank(record['path'], saved['manifest']['data']['train_layouts'])
        eligible = [k for k, d in bank.episodes.items() if len(d['actions']) >= 5]
        selected = rng.sample(eligible, min(4, len(eligible)))
        counts = Counter()
        examples = []
        for name in selected:
            d = bank.episodes[name]
            starts = list(range(0, len(d['actions']), 10))
            for start in starts[:2]:
                if not d['probe_valid'][start]:
                    continue
                begin = max(0, start-10)
                history = torch.tensor(d['codes'][None, begin:start+1], device='cuda').long()
                previous = torch.tensor([[int(d['actions'][j-1]) if j else 17
                                          for j in range(begin, start+1)]], device='cuda')
                base = dict(task=torch.tensor([int(d['task'])], device='cuda'), history=history,
                            previous_actions=previous, start=start, episode_ids=[name])
                raw, available = features.transitions(base, bank.directory)
                p = model.prepare(base['task'], history, previous,
                                  torch.tensor([history.shape[1]], device='cuda'), start=start,
                                  boundary_features=raw, boundary_available=available)
                probabilities = model.rows(p)[0]
                table = probabilities.argmax(-1).cpu().numpy()
                grouped = defaultdict(Counter)
                positions = 0
                for t in range(start, min(start+10, len(d['actions']))):
                    if not d['probe_valid'][t]:
                        continue
                    batch = dict(base, prefix_codes=torch.tensor(d['codes'][None, start:t+1], device='cuda').long(),
                                 successors=torch.tensor(d['successors'][None, t], device='cuda').long())
                    before, after = branch_states(model, p, batch)
                    s = int(before.argmax(-1))
                    targets = after[0].argmax(-1).cpu().numpy()
                    positions += 1
                    counts['positions'] += 1
                    counts['unique_successor_states_sum'] += len(set(targets.tolist()))
                    counts['unique_successor_images_sum'] += len({tuple(x) for x in d['successors'][t].tolist()})
                    for action, target in enumerate(targets):
                        grouped[s, action][int(target)] += 1
                        counts['transitions'] += 1
                        counts['correct'] += int(table[s, action] == target)
                        counts['state_self_loop'] += int(s == target)
                        counts['image_unchanged'] += bool(np.array_equal(d['codes'][t], d['successors'][t, action]))
                        if action:
                            counts['motion_transitions'] += 1
                            counts['motion_correct'] += int(table[s, action] == target)
                if not positions:
                    continue
                counts['programs'] += 1
                for (s, a), outcomes in grouped.items():
                    counts['distinct_premises'] += 1
                    counts['empirical_deterministic_correct_upper_bound'] += max(outcomes.values())
                    if len(outcomes) > 1:
                        counts['ambiguous_premises'] += 1
                        counts['ambiguous_transition_examples'] += sum(outcomes.values())
                        if len(examples) < 5:
                            examples.append(dict(episode=name, start=start, source_state=s, action=a,
                                                 successor_counts=dict(outcomes)))
        n = max(counts['transitions'], 1)
        result = dict(bank=bank.directory.name, counts=dict(counts), accuracy=counts['correct']/n,
                      motion_accuracy=counts['motion_correct']/max(counts['motion_transitions'], 1),
                      deterministic_upper_bound=counts['empirical_deterministic_correct_upper_bound']/n,
                      ambiguity_examples=examples)
        totals.update(counts)
        details.append(result)
        print(json.dumps(result), flush=True)
    n = max(totals['transitions'], 1)
    result = dict(checkpoint=args.checkpoint, training_only=True, optimizer_updates=0, new_rollouts=0,
                  counts=dict(totals), accuracy=totals['correct']/n,
                  deterministic_upper_bound=totals['empirical_deterministic_correct_upper_bound']/n,
                  mean_distinct_states_per_17_outcomes=totals['unique_successor_states_sum']/totals['positions'],
                  mean_distinct_images_per_17_outcomes=totals['unique_successor_images_sum']/totals['positions'],
                  banks=details)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print('TOTAL', json.dumps({k:v for k,v in result.items() if k != 'banks'}), flush=True)


if __name__ == '__main__':
    main()
