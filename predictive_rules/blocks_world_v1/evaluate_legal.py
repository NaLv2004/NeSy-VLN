"""Evaluate sealed Blocks checkpoint with an empirical legal-action mask.

The mask is constructed from training RGB and recorded executed actions only,
before reading test queries. No true stack labels or simulator preconditions
are used to construct it. Generated transitions/goals and checkpoint weights
are unchanged. The physical transition function remains evaluation-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from run import (ROOT, DATA, OLD, BlocksRuleWorld, FrozenVision, features,
                 quantize, read, write, sha, scoring_transition, export_program,
                 NeuralBoundaryGate)
from legal_plan import plan_legal


@torch.no_grad()
def build_applicability(model, frontend, centers, out, minimum_support):
    rows = read(DATA / 'transitions.json')
    names = sorted({r[k] for r in rows for k in ['image', 'next_image']})
    lookup = {n: i for i, n in enumerate(names)}
    raw = features(frontend, names)
    codes = quantize(raw, centers).cuda()
    raw = raw.cuda()
    encoded = []
    for ids in torch.arange(len(names), device='cuda').split(64):
        program = model.prepare_pair(codes[ids], raw[ids], codes[ids], raw[ids])
        encoded.extend(model.ground(program, codes[ids]).argmax(-1).flatten().cpu().tolist())
    counts = np.zeros((512, 19), dtype=np.int64)
    for row in rows:
        counts[encoded[lookup[row['image']]], row['action'] + 1] += 1
    # Any observed image is a valid unchanged endpoint for STOP, as in SFT.
    for state in encoded:
        counts[state, 0] += 1
    applicable = counts >= minimum_support
    np.savez_compressed(out / 'applicability.npz', applicable=applicable, counts=counts)
    write(out / 'applicability_manifest.json', dict(
        source='training RGB -> frozen current state encoder; observed opaque action IDs',
        training_transitions=len(rows), training_images=len(names),
        observed_state_ids=sorted(set(encoded)), minimum_support=minimum_support,
        permitted_real_state_action_pairs=int(applicable[:, 1:].sum()),
        all_observed_real_pairs=int((counts[:, 1:] > 0).sum()),
        no_test_images_or_queries_used=True, no_physical_state_labels_used=True,
        no_simulator_preconditions_used=True, no_old_symbolic_rules_used=True,
        training_rows_sha256=sha(DATA / 'transitions.json'),
        mask_sha256=sha(out / 'applicability.npz'),
        caveat='Training-evidence applicability, not an oracle guarantee if visual state encoding is wrong.'))
    del raw, codes
    torch.cuda.empty_cache()
    return applicable


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--minimum-support', type=int, default=5)
    args = parser.parse_args()
    source = Path(args.run)
    out = Path(args.out)
    assert source.resolve().is_relative_to(ROOT) and out.resolve().is_relative_to(source)
    assert not out.exists(), 'Use a new output folder; preserve prior results'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in {'3', '4', '5', '7'}
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = source / 'latest.pt'
    checkpoint_hash = sha(checkpoint)
    assert checkpoint_hash == read(checkpoint.with_suffix('.COMMITTED.json'))['sha256']
    model = BlocksRuleWorld().cuda().eval()
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(saved['model'])
    model.requires_grad_(False)
    del saved
    adapter = torch.load(source / 'vision_adapter.pt', map_location='cuda', weights_only=False)
    frontend = FrozenVision(adapter['vision_config'])
    print(json.dumps(dict(stage='construct_training_only_action_mask')), flush=True)
    mask = build_applicability(model, frontend, adapter['centers'], out, args.minimum_support)
    # Seal evidence before accessing the same fixed test protocol.
    mask_hash = sha(out / 'applicability.npz')
    queries_path = OLD / 'symbolic/strict_queries.json'
    assert sha(queries_path) == read(source / 'split_audit.json')['queries_sha256']
    queries = read(queries_path)
    names = sorted({q[k] for q in queries for k in ['start_image', 'goal_image']})
    lookup = {n: i for i, n in enumerate(names)}
    raw = features(frontend, names)
    codes = quantize(raw, adapter['centers']).cuda()
    raw = raw.cuda()
    transition = scoring_transition()
    gate = NeuralBoundaryGate([model])
    results = []
    began = time.time()
    for begin in range(0, len(queries), 8):
        part = queries[begin:begin + 8]
        starts = torch.tensor([lookup[q['start_image']] for q in part], device='cuda')
        goals = torch.tensor([lookup[q['goal_image']] for q in part], device='cuda')
        with gate.boundary(0):
            generated = model.generate_pair(codes[starts], raw[starts], codes[goals], raw[goals])
            initial = model.ground(generated, codes[starts]).argmax(-1).cpu().numpy()
        for i, query in enumerate(part):
            fields = {'definitions', 'transitions', 'subgoal_logits', 'start', 'context_raw'}
            program = export_program({k: v for k, v in generated.items() if k in fields}, i)
            plan = plan_legal(program, initial[i], mask)
            proof = []
            state = int(initial[i, 0])
            for action in plan['actions']:
                assert mask[state, action]
                after = int(program.transitions[state, action].argmax())
                proof.append([state, action, after])
                state = after
            assert not plan['found'] or state == int(program.subgoal[0])
            # Environment state enters only the execution scorer, after the
            # complete plan is immutable. No legal oracle is fed back to it.
            actual = tuple(query['start'])
            actual_states = [list(actual)]
            invalid = False
            for action in plan['actions']:
                if action == 0:
                    break
                after = transition(actual, action - 1)
                if after is None:
                    invalid = True
                    break
                actual = after
                actual_states.append(list(actual))
            success = bool(plan['found'] and not invalid and actual == tuple(query['goal']))
            results.append(dict(query_id=query['query_id'], shortest_horizon=query['shortest_horizon'],
                                initial_symbol=initial[i].tolist(), generated_goal=program.subgoal.tolist(),
                                plan=plan, proof=proof, actual_states=actual_states,
                                invalid_action=invalid, success=success, neural_calls_during_execution=0))
        del generated
        if begin % 160 == 0:
            print(json.dumps(dict(stage='legal_mask_evaluation', completed=len(results), total=len(queries))), flush=True)
    gate.close()
    assert sha(checkpoint) == checkpoint_hash and sha(out / 'applicability.npz') == mask_hash
    groups = {}
    for horizon in sorted({r['shortest_horizon'] for r in results}):
        rows = [r for r in results if r['shortest_horizon'] == horizon]
        successes = sum(r['success'] for r in rows)
        groups[str(horizon)] = dict(tasks=len(rows), successes=successes, success_rate=successes / len(rows))
    summary = dict(tasks=len(results), successes=sum(r['success'] for r in results),
                   success_rate=sum(r['success'] for r in results) / len(results), by_horizon=groups,
                   invalid_action_tasks=sum(r['invalid_action'] for r in results),
                   no_plan_tasks=sum(not r['plan']['found'] for r in results),
                   boundary_violations=gate.violations, seconds=time.time() - began,
                   checkpoint_sha256=checkpoint_hash, applicability_sha256=mask_hash,
                   weights_and_generated_transitions_unchanged=True,
                   every_planned_action_checked_against_applicability=True,
                   baseline=read(source / 'after_rules/summary.json'),
                   applicability=read(out / 'applicability_manifest.json'))
    write(out / 'episodes.json', results)
    write(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
