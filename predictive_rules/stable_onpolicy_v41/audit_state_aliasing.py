"""Train-only empirical state-alias audit of the immutable v40 collector.

No simulator, GPU, policy modification, or semantic state labels. All target
codes are recomputed from recorded visual effects using the pinned encoder.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from data import ROOT, sha
from model import FactoredWorld, Config


def summarize(groups):
    samples = sum(sum(counts.values()) for counts in groups.values())
    conflicting = [counts for counts in groups.values() if len(counts) > 1]
    repeats = [counts for counts in groups.values() if sum(counts.values()) > 1]
    return dict(samples=samples, keys=len(groups),
        contradictory_keys=len(conflicting),
        samples_in_contradictory_keys=sum(sum(c.values()) for c in conflicting),
        empirical_deterministic_accuracy_ceiling=sum(max(c.values()) for c in groups.values()) / max(1, samples),
        conditional_entropy_bits=sum(sum(-v * math.log2(v / sum(c.values())) for v in c.values()) for c in groups.values()) / max(1, samples),
        repeated_keys=len(repeats),
        repeated_key_accuracy_ceiling=sum(max(c.values()) for c in repeats) / max(1, sum(sum(c.values()) for c in repeats)))


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    began = time.time()
    cache = ROOT / 'onpolicy_successors_v40'
    commit = json.loads((cache / 'policy.COMMITTED.json').read_text())
    assert sha(cache / 'policy.pt') == commit['sha256']
    saved = torch.load(cache / 'policy.pt', map_location='cpu', weights_only=False)
    model = FactoredWorld(Config(**saved['model_config'])).eval()
    model.load_state_dict(saved['model'], strict=True)
    c = model.cfg
    groups = {name: defaultdict(Counter) for name in ('program_joint_action', 'program_observation_action', 'episode_joint_action', 'global_joint_action')}
    memory_counts = Counter(); late_memory_counts = Counter(); observation_counts = Counter()
    same_initial = []; changed_memory = []; changed_observation = []
    visual_grounding_changed = []; outcome_grounding_variants = []
    episodes = []; examples = []
    for path in sorted((cache / 'counterfactual_onpolicy').glob('*.json')):
        record = json.loads(path.read_text())
        assert record['training_only'] and not record['goal_supervision']
        assert record['source_checkpoint_sha256'] == commit['sha256']
        with np.load(path.with_suffix('.npz')) as archive:
            raw = archive['features']
            future = archive['successor_features']
        current = model.observe(torch.from_numpy(raw)).argmax(-1).numpy()
        flat = future.reshape(-1, 49, 128)
        outcome = torch.cat([model.observe(torch.from_numpy(flat[i:i+128])).argmax(-1) for i in range(0, len(flat), 128)]).numpy().reshape(len(raw), c.actions)
        steps = record['steps']
        if not steps:
            continue
        first_memory = steps[0]['memory']
        for j, step in enumerate(steps):
            t, m, o = step['step'], step['memory'], step['observed']
            assert int(current[t]) == o, 'Pinned encoder differs from collector'
            program = step['proof']['program_sha256']
            memory_counts[m] += 1; observation_counts[o] += 1
            if t >= 5:
                late_memory_counts[m] += 1
                same_initial.append(m == first_memory)
            if j:
                changed_memory.append(m != steps[j-1]['memory'])
                changed_observation.append(o != steps[j-1]['observed'])
            for action, target in enumerate(outcome[t]):
                for name, key in (
                    ('program_joint_action', (program, m, o, action)),
                    ('program_observation_action', (program, o, action)),
                    ('episode_joint_action', (record['episode'], m, o, action)),
                    ('global_joint_action', (m, o, action))):
                    groups[name][key][int(target)] += 1
            ms = F.one_hot(torch.full((c.actions,), m), c.memories).float()
            os = F.one_hot(torch.full((c.actions,), o), c.observations).float()
            ns = F.one_hot(torch.as_tensor(outcome[t].copy()), c.observations).float()
            actions = torch.arange(c.actions)
            # Diagnostic of the PROPOSED arriving-evidence grounder. It is not
            # substituted into the actual policy or claimed as its state.
            before = model.memory_step(ms, os, actions)[0].argmax(-1)
            grounded = model.memory_step(ms, ns, actions)[0].argmax(-1)
            visual_grounding_changed.extend((before != grounded).tolist())
            outcome_grounding_variants.append(int(grounded.unique().numel()))
        episodes.append(dict(episode=record['episode'], steps=len(steps),
            distinct_memory_ids=len({s['memory'] for s in steps}),
            distinct_observation_ids=len({s['observed'] for s in steps})))
        if len(episodes) % 20 == 0:
            print(json.dumps(dict(completed_episodes=len(episodes), seconds=time.time()-began)), flush=True)
    for key, counts in groups['program_joint_action'].items():
        if len(counts) > 1 and len(examples) < 12:
            examples.append(dict(program=key[0], memory=key[1], observed=key[2], action=key[3], successor_counts=dict(counts)))
    result = dict(training_only=True, source_checkpoint=commit, episodes=len(episodes),
        executed_positions=sum(r['steps'] for r in episodes),
        evidence='All 17 recorded action effects at actual policy visits; no physical state or collision labels',
        groups={name: summarize(values) for name, values in groups.items()},
        memory_counts=dict(memory_counts), late_memory_counts=dict(late_memory_counts),
        observation_counts=dict(observation_counts),
        memory_unchanged_from_episode_start_after_five=sum(same_initial)/max(1,len(same_initial)),
        adjacent_memory_change_fraction=sum(changed_memory)/max(1,len(changed_memory)),
        adjacent_observation_change_fraction=sum(changed_observation)/max(1,len(changed_observation)),
        episodes_with_single_memory=sum(r['distinct_memory_ids']==1 for r in episodes),
        proposed_arriving_evidence_grounder=dict(
            target_changed_from_using_current_observation=sum(visual_grounding_changed)/max(1,len(visual_grounding_changed)),
            positions_with_multiple_action_outcome_memories=sum(v>1 for v in outcome_grounding_variants),
            positions=len(outcome_grounding_variants)),
        contradictory_examples=examples, episode_rows=episodes, seconds=time.time()-began,
        limitations='Empirical ceilings concern deterministic predictions under fixed codes and fixed conditioning; stochastic rules can represent ambiguity. Low aliasing in this finite cache does not prove sufficient state. A constant memory may encode useful scene context; these statistics alone do not justify adding semantic priors.')
    Path(__file__).with_name('state_aliasing_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('episode_rows','contradictory_examples','memory_counts','late_memory_counts','observation_counts')}))


if __name__ == '__main__':
    main()
