"""Actual navigation comparison on the unchanged scene-isolated splits.

Images may be recorded at any time. Only revision boundaries expose them to
the new policy. The controller accepts neither observations nor feedback.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image

from predictive_nesy.data import read
from predictive_nesy.environment import FrozenVision, load_previous, make_bank
from predictive_nesy.interpreter import BlockController, export_program
from predictive_nesy.model import GeneratedWorldModel, ModelConfig
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.belief_interpreter import BeliefController


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        for key in ['domain', 'distance_group', 'category']:
            groups[key + ':' + row[key]].append(row)
        if row['domain'] == 'normal' and row['distance_group'] in ['near', 'medium']:
            groups['normal_near_medium'].append(row)
    statistics = {key:dict(episodes=len(v), successes=sum(x['success'] for x in v),
        success_rate=sum(x['success'] for x in v)/len(v),
        collision_action_rate=sum(x['collisions'] for x in v)/max(sum(x['steps'] for x in v), 1)) for key,v in groups.items()}
    return dict(**statistics['all'], by_group=statistics,
        mean_steps=float(np.mean([r['steps'] for r in rows])),
        vision_policy_calls=sum(r['vision_policy_calls'] for r in rows),
        neural_calls_outside_boundaries=sum(r['neural_calls_outside_boundaries'] for r in rows),
        rule_step_seconds=sum(r['rule_step_seconds'] for r in rows),
        boundary_seconds=sum(r['boundary_seconds'] for r in rows))


@torch.no_grad()
def run(model, frontend_source, bank, out, suite, *, previous=False, beam_width=32, cached_rows=None):
    anonymous=getattr(model,'execution_schema',None)=='anonymous_v1'
    if anonymous:
        from predictive_nesy.anonymous_interpreter import Controller as AnonymousController, export_program as export_anonymous
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = bank.dev if suite == 'development' else bank.fixed
    assert len(episodes) == 80
    train_scenes = set(bank.provenance['train_layouts'])
    assert not {e['scene'] for e in episodes}.intersection(train_scenes)
    atomic(out/'manifest.json', dict(suite=suite, episode_ids=[e['episode_id'] for e in episodes],
        data=bank.provenance, horizon=32, revision_interval=10, previous_reactive_model=previous,
        decoding='greedy', planner=None if previous else 'beam10_width'+str(beam_width)))
    vision = FrozenVision(frontend_source)
    model.eval()
    gate = None if previous else NeuralBoundaryGate([model, vision.vision])
    rows = list(cached_rows or [])
    cached_ids = {r['episode_id'] for r in rows}
    pending = [e for e in episodes if e['episode_id'] not in cached_ids]
    sims = []
    started = time.time()
    try:
        sims = [bank.simulator(out/f'sim_{i}.log', 0, evaluation=True) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0, len(pending), 4):
                batch = pending[offset:offset+4]
                responses = list(pool.map(lambda item:item[0].call('reset', episode=item[1]), zip(sims,batch)))
                image_histories = [[Image.fromarray(np.asarray(r['rgb'])[...,:3].astype(np.uint8))] for r in responses]
                action_histories = [[17] for _ in batch]
                controller_type=AnonymousController if anonymous else BeliefController if (not previous and hasattr(model.cfg,'observed_factors')) else BlockController
                controllers = [controller_type() for _ in batch]
                records = [dict(episode=e, steps=[], programs=[], metrics={}, vision_policy_calls=0,
                    neural_calls_outside_boundaries=0, boundary_seconds=0., rule_step_seconds=0.) for e in batch]
                task = torch.tensor([['chair','table'].index(e['category']) for e in batch], device='cuda')
                active = list(range(len(batch)))
                old_history = []
                if previous:
                    _, memory = model.initial_state(len(batch), 'cuda')
                    previous_action = torch.full_like(task, 17)
                for step in range(32):
                    if previous:
                        neural_start = time.perf_counter()
                        codes = vision.encode([history[-1] for history in image_histories])
                        old_history.append(codes)
                        if step % 10 == 0:
                            old_program = model.generate(task, torch.stack(old_history[-11:],1), step)
                        result = model.step(old_program, codes, memory, previous_action, step)
                        chosen = result['probabilities'].argmax(-1)
                        memory, previous_action = result['state_values'], chosen
                        neural_time = time.perf_counter() - neural_start
                        for i in active:
                            records[i]['vision_policy_calls'] += 1
                            records[i]['boundary_seconds'] += neural_time / len(batch)
                            records[i]['steps'].append(dict(step=step, action=int(chosen[i]), state=result['state_ids'][i].tolist()))
                    else:
                        if step % 10 == 0:
                            neural_start = time.perf_counter()
                            width = min(step+1, 11)
                            with gate.boundary(step):
                                history = torch.stack([vision.encode(image_histories[i][-width:]) for i in active])
                                previous_actions = torch.tensor([action_histories[i][-width:] for i in active], device='cuda')
                                lengths = torch.full((len(active),), width, dtype=torch.long, device='cuda')
                                generated = model.generate(task[active], history, previous_actions, lengths, start=step)
                                states = model.ground(generated, history[:,-1]).argmax(-1).cpu().numpy()
                            for j, i in enumerate(active):
                                program = export_anonymous(generated,j) if anonymous else export_program(generated, j)
                                plan = controllers[i].revise(program, states[j], horizon=min(10,32-step), beam_width=beam_width)
                                program_path = out/batch[i]['episode_id']/f'program_{step:02d}.npz'
                                program_path.parent.mkdir(parents=True, exist_ok=True)
                                program.save(program_path)
                                records[i]['programs'].append(dict(start=step, initial_state=states[j].tolist(), plan=plan,
                                    file=program_path.name, observed_frame_indices=list(range(max(0,step-10),step+1))))
                                records[i]['vision_policy_calls'] += 1
                            boundary_time = time.perf_counter() - neural_start
                            for i in active:
                                records[i]['boundary_seconds'] += boundary_time / len(active)
                            del generated
                        chosen = torch.zeros(len(batch), dtype=torch.long)
                        for i in active:
                            rule_start = time.perf_counter()
                            action, proof = controllers[i].next_action()
                            records[i]['rule_step_seconds'] += time.perf_counter()-rule_start
                            chosen[i] = action
                            records[i]['steps'].append(proof)
                    requests = [(i,int(chosen[i])) for i in active]
                    responses = list(pool.map(lambda item:sims[item[0]].call('step', action=item[1]), requests))
                    remaining = []
                    for i, response in zip(active, responses):
                        image = Image.fromarray(np.asarray(response['rgb'])[...,:3].astype(np.uint8))
                        # Raw recording is not passed to controller.next_action.
                        image_histories[i].append(image)
                        action_histories[i].append(int(chosen[i]))
                        records[i]['steps'][-1]['collision_for_report_only'] = bool(response.get('feedback',{}).get('collision',False))
                        records[i]['metrics'] = response.get('metrics',{})
                        if not response['done']:
                            remaining.append(i)
                    active = remaining
                    if not active:
                        break
                for i, (episode, record) in enumerate(zip(batch,records)):
                    folder = out/episode['episode_id']
                    folder.mkdir(exist_ok=True)
                    image_histories[i][0].save(folder/'first.png')
                    image_histories[i][-1].save(folder/'last.png')
                    atomic(folder/'trace.json', record)
                    row = dict(episode_id=episode['episode_id'], scene=episode['scene'],
                        domain=episode['domain'], distance_group=episode['distance_group'], category=episode['category'],
                        success=bool(record['metrics'].get('success',False)), steps=len(record['steps']),
                        collisions=sum(r['collision_for_report_only'] for r in record['steps']),
                        **{key:record[key] for key in ['vision_policy_calls','neural_calls_outside_boundaries','boundary_seconds','rule_step_seconds']})
                    rows.append(row)
                    atomic(folder/'evaluation.json', row)
                atomic(out/'status.json', dict(state='evaluating', completed=len(rows), total=len(episodes), time=time.time()))
                print(json.dumps(dict(evaluation=suite, previous=previous, completed=len(rows), successes=sum(r['success'] for r in rows))), flush=True)
        summary = dict(summarize(rows), seconds=time.time()-started, suite=suite, previous=previous)
        summary['reused_completed_episodes'] = len(cached_ids)
        if gate is not None:
            assert gate.violations == 0
            summary['boundary_gate_neural_module_calls'] = gate.calls
            summary['boundary_gate_violations'] = gate.violations
        atomic(out/'summary.json', summary)
        atomic(out/'status.json', dict(state='complete', time=time.time(), summary=summary))
        return summary
    finally:
        if gate is not None:
            gate.close()
        for simulator in sims:
            simulator.close()
        del vision
        torch.cuda.empty_cache()


def evaluate_previous(model, manifest, commit, bank, out, suite):
    from predictive_nesy.data import BASE
    cached=[]
    source=BASE/'eval_normal_short_medium_20261014'/suite
    if (source/'summary.json').exists():
        summary=read(source/'summary.json')
        assert summary['checkpoint_sha256']==commit['checkpoint_sha256']
        for e in (bank.dev if suite=='development' else bank.fixed):
            path=source/e['episode_id']/'evaluation.json'
            if path.exists():
                row=read(path)
                assert row['checkpoint_sha256']==commit['checkpoint_sha256']
                row.update(vision_policy_calls=row['steps'], neural_calls_outside_boundaries=0,
                           boundary_seconds=0.,rule_step_seconds=0.,reused=True)
                cached.append(row)
    result = run(model, manifest['source_commit'], bank, out, suite, previous=True, cached_rows=cached)
    atomic(Path(out)/'checkpoint.json', commit)
    return result


def evaluate_generated(model, manifest, bank, out, suite):
    return run(model, manifest['frozen_vision'], bank, out, suite)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--suite', choices=['development','fixed80'], default='development')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'3','4','5','7'}:
        raise RuntimeError('Use one explicitly authorized GPU')
    checkpoint=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
    if checkpoint['manifest'].get('architecture','').startswith('anonymous_generated_rules_'):
        from predictive_nesy.anonymous_model import AnonymousRuleModel,Config
        model=AnonymousRuleModel(Config(**checkpoint['model_config'])).cuda()
    elif 'observed_factors' in checkpoint['model_config']:
        from predictive_nesy.model_memory import PredictiveMemoryModel, MemoryConfig
        model=PredictiveMemoryModel(MemoryConfig(**checkpoint['model_config'])).cuda()
    else:
        model=GeneratedWorldModel(ModelConfig(**checkpoint['model_config'])).cuda()
    model.load_state_dict(checkpoint['model'])
    evaluate_generated(model,checkpoint['manifest'],make_bank(),a.out,a.suite)


if __name__=='__main__':
    main()
