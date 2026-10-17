"""Single-free-GPU training; latest/best checkpoints and resumable progress."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch

from predictive_nesy.data import BASE, load_training_windows, batch_windows, read, sha
from predictive_nesy.environment import load_previous, make_bank
from predictive_nesy.model import GeneratedWorldModel, ModelConfig
from predictive_nesy.training import objective


def atomic(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def save_checkpoint(path, model, optimizer, update, manifest, rng):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
        model_config=model.export_config(), manifest=manifest,
        random_state=rng.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic(path.with_suffix('.COMMITTED.json'), dict(path=path.name, sha256=sha(path), update=update, time=time.time()))


@torch.no_grad()
def learned_symbol_targets(teacher, windows, out):
    from nesy_vla.observation_inverse import observation_equalities
    path = out / 'learned_symbol_targets.npz'
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if len(data['labels']) != len(windows):
            raise ValueError('Cached target count changed')
        for w, value in zip(windows, data['labels']):
            w['teacher_ids'] = value.astype(np.int64)
        return
    labels = np.zeros((len(windows), 11, 4), dtype=np.uint8)
    for start in sorted({w['start'] for w in windows}):
        indices = [j for j, w in enumerate(windows) if w['start'] == start]
        for offset in range(0, len(indices), 64):
            chosen = indices[offset:offset+64]
            batch = batch_windows([windows[i] for i in chosen], 'cuda')
            width = min(start + 1, 11)
            program = teacher.generate(batch['task'], batch['history'][:, :width], start)
            ids = []
            for t in range(11):
                # A fixed boundary definition set grounds future supervision.
                values = observation_equalities(teacher, program, batch['targets'][:, t], start)
                ids.append(values.argmax(-1))
            labels[chosen] = torch.stack(ids, 1).cpu().numpy()
    np.savez_compressed(path, labels=labels)
    for w, value in zip(windows, labels):
        w['teacher_ids'] = value.astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-baseline', action='store_true')
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '7':
        raise RuntimeError('This experiment is explicitly restricted to physical GPU 7')
    out = Path(args.out).resolve()
    if not out.is_relative_to(BASE / 'predictive_rules_20261015'):
        raise ValueError('Output must remain in the isolated experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'train.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(1510)
    np.random.seed(1510)
    rng = random.Random(1510)
    started = time.time()
    atomic(out / 'status.json', dict(state='preparing', pid=os.getpid(), time=started, gpu=7))
    teacher, old_manifest, old_commit = load_previous()
    bank = make_bank()
    assert bank.provenance == old_manifest['data']
    windows, provenance = load_training_windows(BASE / 'perceptual_sequences_v73/tail_1e4', bank.provenance['train_layouts'])
    manifest = dict(architecture='generated_predictive_rule_tables_v1', model=vars(ModelConfig()),
        source_model=old_commit, data=provenance, frozen_vision=old_manifest['source_commit'],
        rule_revision_interval=10, horizon=32, neural_calls_between_revisions=0,
        generated_parameters=['definitions', 'parents', 'transitions', 'risks', 'goals', 'values', 'weights'],
        runtime_state_update='generated_table(current_state, executed_action)',
        planning='numpy_beam_search_depth10_width32',
        direct_action_policy=False, maps=False, future_observations='training_targets_only',
        symbol_supervision='previous_learned_observation_IDs_under_boundary_definitions',
        source_neural_policy_is_runtime_dependency=False,
        train_args=vars(args), created=time.time())
    if (out/'manifest.json').exists():
        previous = read(out/'manifest.json')
        assert previous['source_model'] == old_commit and previous['data'] == provenance
        manifest = previous
    else:
        atomic(out/'manifest.json', manifest)
    learned_symbol_targets(teacher, windows, out)
    model = GeneratedWorldModel().cuda()
    # Transfer only the evidence encoder, never the old reactive transition or
    # direct action table. Every executable world-model table is generated anew.
    model.tokens.load_state_dict(teacher.tokens.state_dict())
    model.observation.load_state_dict(teacher.observation.state_dict())
    model.context_read.load_state_dict(teacher.context_read.state_dict())
    model.history.load_state_dict(teacher.history.state_dict())
    model.task.load_state_dict(teacher.task.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    update = 0
    if args.resume and (out/'latest.pt').exists():
        checkpoint = torch.load(out/'latest.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        update = checkpoint['update']
        rng.setstate(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
        del checkpoint
    if not args.skip_baseline:
        from evaluate import evaluate_previous
        for suite in ['development', 'fixed80']:
            folder = out/'baseline_1097'/suite
            if not (folder/'summary.json').exists():
                atomic(out/'status.json', dict(state='measuring_reference', suite=suite, pid=os.getpid(), time=time.time(), gpu=7))
                evaluate_previous(teacher, old_manifest, old_commit, bank, folder, suite)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    by_start = {s:[w for w in windows if w['start']==s] for s in sorted({w['start'] for w in windows})}
    start_choices = [w['start'] for w in windows]
    best = read(out/'best_development.json')['successes'] if (out/'best_development.json').exists() else -1
    while update < args.updates:
        step_started = time.time()
        model.train()
        start = rng.choice(start_choices)
        chosen = rng.choices(by_start[start], k=args.batch_size)
        batch = batch_windows(chosen, 'cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = objective(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite objective; committed checkpoint preserved')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite gradient; committed checkpoint preserved')
        optimizer.step()
        update += 1
        torch.cuda.synchronize()
        metrics.update(update=update, seconds=time.time()-step_started, elapsed_seconds=time.time()-started,
            grad_norm=float(norm), boundary=start, batch_size=args.batch_size,
            memory_peak_gib=torch.cuda.max_memory_allocated()/2**30, time=time.time())
        with (out/'train.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
        atomic(out/'status.json', dict(state='training', pid=os.getpid(), gpu=7, **metrics))
        if update == 1 or update % 20 == 0:
            print(json.dumps(metrics), flush=True)
        if update % args.checkpoint_every == 0 or update == args.updates:
            save_checkpoint(out/'latest.pt', model, optimizer, update, manifest, rng)
        if update % args.eval_every == 0:
            from evaluate import evaluate_generated
            atomic(out/'status.json', dict(state='evaluating_development', update=update, pid=os.getpid(), gpu=7, time=time.time()))
            report = evaluate_generated(model, manifest, bank, out/f'development_step_{update:06d}', 'development')
            report['update'] = update
            with (out/'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(report, allow_nan=False)+'\n')
            if report['successes'] > best:
                best = report['successes']
                save_checkpoint(out/'best_development.pt', model, optimizer, update, manifest, rng)
                atomic(out/'best_development.json', report)
    atomic(out/'status.json', dict(state='training_budget_complete', update=update, pid=os.getpid(), gpu=7,
        time=time.time(), goal_complete=False, next_action='Evaluate learning curves and continue toward measured reference parity'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
