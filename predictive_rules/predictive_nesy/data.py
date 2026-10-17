"""Causal revision windows; future images are labels, never generator inputs."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import torch


BASE = Path("/home/user/xjy/nesy_vla_20261010")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_training_windows(directory, train_layouts, interval=10, horizon=32):
    directory = Path(directory)
    rows = read(directory / "training_rows.json")
    codes = np.load(directory / "visual_tokens.npy")
    manifest = read(directory / "manifest.json")
    if manifest["data_provenance"]["train_layouts"] != sorted(train_layouts):
        raise ValueError("The training-only corpus layout provenance differs")
    episodes = defaultdict(list)
    outcomes_path = Path('/home/user/xjy/strips_wm_navigation_20261009/data/search_cost_supervision_v3/execution_outcomes.json')
    outcome_root = outcomes_path.parents[1]
    outcomes = {str((outcome_root / r['image']).resolve()): r for r in read(outcomes_path)}
    for index, row in enumerate(rows):
        match = re.search(r'P25_train_(\d+)_', row.get('source_episode_id', row['episode_id']))
        if not match or 'layout_' + match.group(1) not in train_layouts:
            raise ValueError('A source row is outside the permitted training layouts: ' + row['episode_id'])
        episodes[row["episode_id"]].append(index)
    windows = []
    for episode, indices in episodes.items():
        # Both original and recovery IDs retain their source layout number.
        # Corpus provenance is additionally audited against original manifests.
        category = {rows[i]["category"] for i in indices}
        if len(category) != 1:
            raise ValueError("A trajectory changes its goal unexpectedly")
        task = ["chair", "table"].index(next(iter(category)))
        actions = [int(rows[i]["label"]) for i in indices]
        if actions[-1] != 0:
            raise ValueError("A successful source demonstration must terminate in STOP")
        for start in range(0, min(len(indices), horizon), interval):
            history_indices = indices[max(0, start - interval):start + 1]
            history = np.zeros((interval + 1, codes.shape[1]), dtype=np.int64)
            history[:len(history_indices)] = codes[history_indices]
            previous = np.full(interval + 1, 17, dtype=np.int64)
            offset = max(0, start - interval)
            previous[:len(history_indices)] = [actions[t - 1] if t else 17 for t in range(offset, start + 1)]
            future_indices = indices[start:start + interval + 1]
            n = len(future_indices)
            future = np.zeros_like(history)
            future[:n] = codes[future_indices]
            future[n:] = future[n - 1]
            chosen = np.zeros(interval + 1, dtype=np.int64)
            chosen[:n] = actions[start:start + n]
            valid = np.arange(interval + 1) < n
            supervised = np.zeros(interval + 1, dtype=bool)
            supervised[:n] = [rows[i].get("supervised", True) for i in future_indices]
            remaining = np.maximum(len(indices) - 1 - start - np.arange(interval + 1), 0)
            risks = np.full((interval + 1, 17), -1., dtype=np.float32)
            for j, index in enumerate(future_indices):
                record = outcomes.get(str(Path(rows[index]['path']).resolve()))
                if record is not None:
                    risks[j] = np.where(record['execution_valid'], record['observed_collision'], -1.)
                    risks[j, 0] = -1.  # STOP is not a motion/collision target.
            windows.append(dict(episode_id=episode, task=task, start=start, history=history,
                previous_actions=previous, history_length=len(history_indices), targets=future,
                actions=chosen, valid=valid, supervised=supervised,
                goal=valid & (chosen == 0), value=(.97 ** remaining).astype(np.float32), risks=risks))
    provenance = dict(source=str(directory), rows=len(rows), episodes=len(episodes),
        windows=len(windows), rows_sha256=sha(directory / "training_rows.json"),
        codes_sha256=sha(directory / "visual_tokens.npy"),
        source_manifest_sha256=sha(directory / "manifest.json"),
        train_layouts=sorted(train_layouts), interval=interval, horizon=horizon,
        future_frames="supervision_only", boundary_context="current_and_past_only")
    provenance['outcomes_sha256'] = sha(outcomes_path)
    provenance['known_risk_labels'] = sum(int((w['risks'] >= 0).sum()) for w in windows)
    return windows, provenance


def batch_windows(windows, device):
    result = {}
    for name in ("task", "start", "history", "previous_actions", "history_length", "targets", "actions",
                 "valid", "supervised", "goal", "value", "teacher_ids", "risks"):
        if name in windows[0]:
            result[name] = torch.as_tensor(np.asarray([w[name] for w in windows]), device=device)
    result["episode_id"] = [w["episode_id"] for w in windows]
    return result
