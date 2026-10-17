"""Controlled train-only test of two-step probability propagation.

Both copies start with identical v56 rule weights and frozen perception,
definitions, memory prediction and targets. The only difference is whether
the second-step likelihood marginalizes all 64 intermediate observation IDs
or chooses the first-step mode, as the existing recurrent loss does.
No simulator, official validation, action policy, or runtime change is used.
"""
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0, str(ROOT/'src_goal_equivalence_v57'))
from model import FactoredWorld, Config, hard
from data import sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.manual_seed(581017)
    os.nice(10)
    began = time.time()
    out = ROOT/'diagnostic_probability_training_v58'
    out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(), 'Completed pilot must not be repeated'
    source = ROOT/'diagnostic_goal_equivalence_v57/source.pt'
    commit = json.loads(source.with_suffix('.COMMITTED.json').read_text())
    assert sha(source) == commit['sha256']
    saved = torch.load(source, map_location='cpu', weights_only=False)
    model = FactoredWorld(Config(**saved['model_config'])).eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    c = model.cfg
    assert c.observations == 64 and c.memories == 32 and c.actions == 17
    dataset = ROOT/'onpolicy_successors_v54'
    manifest = json.loads((dataset/'manifest.json').read_text())
    assert manifest['training_only'] and manifest['no_goal_or_imitation_supervision']
    paths = sorted((dataset/'counterfactual_onpolicy').glob('*.json'))
    buckets = defaultdict(list)
    for p in paths:
        r = json.loads(p.read_text())
        assert r['training_only'] and r['scene'] in manifest['train_layouts']
        buckets[(r['domain'], r['distance_group'], r['category'])].append(p)
    rng = random.Random(581017)
    holdout = set()
    for key, group in sorted(buckets.items()):
        rng.shuffle(group)
        holdout.update(group[:max(1, len(group)//5)])
    rows = {'fit': [], 'holdout': []}
    episodes = {'fit': [], 'holdout': []}
    with torch.no_grad():
        for num, p in enumerate(paths):
            meta = json.loads(p.read_text())
            assert sha(p.with_suffix('.npz')) == meta['output_sha256']
            with np.load(p.with_suffix('.npz'), allow_pickle=False) as a:
                x = torch.from_numpy(a['features'].copy())[None]
                acts = torch.from_numpy(a['recorded_actions'].copy()).long()
            length = x.shape[1]
            previous = torch.cat([torch.tensor([17]), acts])[:length][None]
            task = torch.tensor([['chair', 'table'].index(meta['category'])])
            contexts = model.context_sequence(x, hard(model.observe(x)), previous, task)
            part = 'holdout' if p in holdout else 'fit'
            episodes[part].append(meta['episode'])
            for j, program in enumerate(meta['programs']):
                boundary = program['start']
                end = meta['programs'][j+1]['start'] if j+1 < len(meta['programs']) else length
                if boundary >= length:
                    continue
                context = contexts[:, boundary]
                lo, evidence, _ = model.observation_encoding(x, context)
                obs = hard(lo)
                lm, _, _ = model.memory_sequence(obs[:, :boundary+1], previous[:, :boundary+1], evidence[:, :boundary+1])
                memory = hard(lm[:, -1])[:, None]
                defs = F.normalize(model.observation_definitions[None] + model.observation_definition_delta(context), dim=-1)
                mdefs = model.definitions(context)
                for t in range(boundary, min(end, length-1, len(acts))):
                    action = acts[t].reshape(1, 1)
                    nm, _ = model.transition(context, memory, obs[:, t:t+1], action)
                    m1 = hard(nm)
                    # Two physical actions and both actual subsequent images,
                    # expressed in this fixed program basis. No action after STOP.
                    if t+2 < length and t+1 < len(acts) and int(acts[t]) != 0:
                        a1 = acts[t+1].reshape(1, 1)
                        rows[part].append(dict(
                            context=context[0].clone(), definitions=defs[0].clone(),
                            memory0=torch.einsum('bam,bmw->baw', memory, mdefs)[0, 0].clone(),
                            memory1=torch.einsum('bam,bmw->baw', m1, mdefs)[0, 0].clone(),
                            obs=obs[0, t].clone(),
                            action0=model.action_embedding(action)[0, 0].clone(),
                            action1=model.action_embedding(a1)[0, 0].clone(),
                            preserve0=(model.operator_class(action).argmax(-1)==1).float()[0, 0],
                            preserve1=(model.operator_class(a1).argmax(-1)==1).float()[0, 0],
                            target1=lo[0, t+1].argmax().clone(), target2=lo[0, t+2].argmax().clone(),
                            source_memory=memory[0, 0].argmax().clone(),
                            source_observation=obs[0, t].argmax().clone(),
                            action_id0=acts[t].clone(), action_id1=acts[t+1].clone(),
                        ))
                    memory = m1
            if num % 20 == 19:
                print(json.dumps(dict(phase='encoding',episodes=num+1,seconds=time.time()-began)), flush=True)
    assert not set(episodes['fit']) & set(episodes['holdout'])
    assert min(map(len, rows.values())) > 100
    data = {part: {k: torch.stack([r[k] for r in values]) for k in values[0]} for part, values in rows.items()}
    del rows

    def forward(head, d):
        n = len(d['obs'])
        p1 = head(d['memory0'][:, None], d['obs'][:, None], d['definitions'],
                  d['action0'][:, None], d['context'], d['preserve0'][:, None, None]).exp()[:, 0]
        p2 = head(d['memory1'][:, None].expand(-1, c.observations, -1),
                  torch.eye(c.observations)[None].expand(n, -1, -1), d['definitions'],
                  d['action1'][:, None].expand(-1, c.observations, -1), d['context'],
                  d['preserve1'][:, None, None].expand(-1, c.observations, -1)).exp()
        # Every operand above is a row conditioned on a hard integer premise.
        # The weighted sum is exact discrete marginalization, never a soft state
        # fed to the neural transition function.
        marginalized = torch.einsum('bi,bij->bj', p1, p2)
        modal = p2[torch.arange(n), p1.detach().argmax(-1)]
        known_second = p2[torch.arange(n), d['target1']]
        return p1, marginalized, modal, known_second

    # Verify a small exact sum against an explicitly exported NumPy table.
    d = {k: v[:1] for k, v in data['holdout'].items()}
    with torch.no_grad():
        _, marginalized, _, _ = forward(model.definition_rule, d)
        table = model.compile(d['context'], torch.tensor([0]))
        root = int(d['source_memory'][0])*c.observations + int(d['source_observation'][0])
        a0, a1 = int(d['action_id0'][0]), int(d['action_id1'][0])
        mid_memory = table['next_memory'][root, a0]
        mid_states = mid_memory*c.observations + np.arange(c.observations)
        np_result = table['outcome_probability'][root, a0].astype(np.float64) @ table['outcome_probability'][mid_states, a1].astype(np.float64)
        error = float(np.max(np.abs(np_result-marginalized[0].numpy())))
        assert error < 2e-5, error
    heads = {name: copy.deepcopy(model.definition_rule).requires_grad_(True) for name in ('modal_control', 'all_branches')}
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4) for name,head in heads.items()}
    start_weights = {name: {k:v.clone() for k,v in h.state_dict().items()} for name,h in heads.items()}
    assert all(torch.equal(start_weights['modal_control'][k],v) for k,v in start_weights['all_branches'].items())
    history = []

    def nll(p, target):
        return -p[torch.arange(len(p)), target].clamp_min(1e-12).log().mean()

    @torch.no_grad()
    def evaluate(update):
        result = {}
        for name, head in heads.items():
            result[name] = {}
            for part, values in data.items():
                sums = defaultdict(float)
                for start in range(0, len(values['obs']), 64):
                    d = {k:v[start:start+64] for k,v in values.items()}
                    p1, total, modal, known = forward(head, d)
                    n = len(p1)
                    measurements = dict(one_step_accuracy=(p1.argmax(-1)==d['target1']).float().mean(),
                        two_step_probability_accuracy=(total.argmax(-1)==d['target2']).float().mean(),
                        two_step_modal_accuracy=(modal.argmax(-1)==d['target2']).float().mean(),
                        known_middle_accuracy=(known.argmax(-1)==d['target2']).float().mean(),
                        one_step_nll=nll(p1,d['target1']),two_step_probability_nll=nll(total,d['target2']),
                        two_step_modal_nll=nll(modal,d['target2']),two_step_confidence=total.max(-1).values.mean())
                    for key,value in measurements.items():sums[key] += float(value)*n
                result[name][part] = {key:value/len(values['obs']) for key,value in sums.items()}
        row = dict(update=update,seconds=time.time()-began,metrics=result)
        history.append(row)
        (out/'progress.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)

    evaluate(0)
    selection = torch.Generator().manual_seed(581018)
    for update in range(1,401):
        selected = torch.randint(len(data['fit']['obs']),(32,),generator=selection)
        batch = {k:v[selected] for k,v in data['fit'].items()}
        for name,head in heads.items():
            optimizers[name].zero_grad(set_to_none=True)
            first,total,modal,known = forward(head,batch)
            loss = nll(first,batch['target1']) + .5*nll(known,batch['target2']) + nll(total if name=='all_branches' else modal,batch['target2'])
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
            assert torch.isfinite(norm)
            optimizers[name].step()
        if update%100==0:evaluate(update)
    record = dict(source=commit,source_path=str(source),time=time.time(),seconds=time.time()-began,
        pid=os.getpid(),device='CPU',torch_threads=2,updates=400,
        head_parameters=sum(p.numel() for p in heads['all_branches'].parameters()),
        positions={k:len(v['obs']) for k,v in data.items()},episodes=episodes,
        history=history,full_table_marginalization_error=error,
        training_only=True,official_dev80_used=False,finalfixed80_used=False,
        live_training_modified=False,new_rollouts=0,representations_and_labels_frozen=True,
        identical_initial_weights_and_batches=True,no_neural_action_policy=True,
        reference='https://proceedings.mlr.press/v97/hafner19a.html',
        adaptation='Exact two-step discrete outcome likelihood, inspired by multi-step latent overshooting; not a PlaNet architecture reproduction.',
        limitation='Source representation has already seen these training episodes. Heldout is disjoint only for this additional rule fit. This is not a navigation success rate.')
    torch.save({name:head.state_dict() for name,head in heads.items()},out/'heads.pt')
    (out/'result.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ('history','episodes')}),flush=True)


if __name__ == '__main__':
    main()
