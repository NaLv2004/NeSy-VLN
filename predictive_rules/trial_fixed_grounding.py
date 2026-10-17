"""Bounded intervention: improve rules while keeping all symbol meanings fixed."""
import copy, hashlib, json, os, random, time
from pathlib import Path
import torch
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel, Config
from predictive_nesy.visual_graph_subgoal_training import objective, transition_objective
from predictive_nesy.subgoal_data import load_windows, batch, TransitionBank

ROOT = Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT = ROOT / 'fixed_grounding_trial'


def main():
    if (OUT / 'candidate.pt').exists():
        raise RuntimeError('This bounded trial has already completed')
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.manual_seed(913)
    rng = random.Random(913)
    source = ROOT / 'run_visual_graph_subgoal_v5/checkpoint_004000.pt'
    saved = torch.load(source, map_location='cpu', weights_only=False)
    model = VisualGraphSubgoalModel(Config(**saved['model_config'])).cuda()
    model.load_state_dict(saved['model'])
    # Freeze the complete visual grounding and its context, so that an easier
    # prediction target cannot be obtained by changing or collapsing state IDs.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('transitions.', 'subgoal.')))
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if not name.startswith(('transitions.', 'subgoal.'))}
    windows, _ = load_windows(saved['manifest']['data']['source'], saved['manifest']['data']['train_layouts'])
    starts = [w['start'] for w in windows]
    by_start = {s: [w for w in windows if w['start'] == s] for s in set(starts)}
    banks = [TransitionBank(ROOT / name, saved['manifest']['data']['train_layouts']) for name in
             ['counterfactual_training_v1', 'counterfactual_own_policy_v1', 'counterfactual_anonymous_v5']]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    manifest = copy.deepcopy(saved['manifest'])
    manifest['fixed_grounding_intervention'] = dict(source=str(source), source_update=saved['update'],
        updates=600, trainable=['transitions', 'subgoal'], all_other_parameters_fixed=True,
        learning_rate=1e-4, training_only=True)
    (OUT / 'launcher.json').write_text(json.dumps(dict(pid=os.getpid(), gpu=7, source=str(Path(__file__).parent), time=time.time())))
    began = time.time()
    for step in range(1, 601):
        start = rng.choice(starts)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data = batch(rng.choices(by_start[start], k=256), 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = objective(model, data)
        loss.backward()
        del loss, data
        data = rng.choice(banks).sample(rng, 16, 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, extra = transition_objective(model, data)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite fixed-grounding gradient')
        optimizer.step()
        del loss, data
        if step % 100 == 0:
            torch.cuda.synchronize()
            metrics.update(extra, step=step, seconds=time.time()-began, gradient_norm=float(norm),
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            with (OUT / 'train.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            print(json.dumps(metrics), flush=True)
    unchanged = all(torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in before.items())
    if not unchanged:
        raise AssertionError('Symbol meanings changed during the fixed-grounding trial')
    manifest['fixed_grounding_intervention']['frozen_parameters_bitwise_unchanged'] = True
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), model_config=model.export_config(),
                    manifest=manifest, update=saved['update']+600), OUT / 'candidate.tmp')
    os.replace(OUT / 'candidate.tmp', OUT / 'candidate.pt')
    commit = dict(path='candidate.pt', sha256=hashlib.sha256((OUT/'candidate.pt').read_bytes()).hexdigest(),
                  update=saved['update']+600, time=time.time())
    (OUT / 'candidate.COMMITTED.json').write_text(json.dumps(commit, indent=2))
    print(json.dumps(dict(completed=True, seconds=time.time()-began, frozen_parameters_bitwise_unchanged=unchanged,
                          checkpoint=commit)), flush=True)


if __name__ == '__main__':
    main()
