"""Train current NeSy rule/goal system on the existing reconstructed Blocks task.

Training reads RGB and opaque actions only. An old image-only FSQ encoder
supplies anonymous representation-distillation targets. Its old dynamics,
operators and planner are never reused. Private simulator states are opened
only after checkpoints have been sealed, for independent execution scoring.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
PAPER = Path('/home/user/xjy/strips_wm_navigation_20261009')
OLD = PAPER / 'runs/paper_blocks_early_layer_e9'
DATA = PAPER / 'runs/paper_blocks_reproduction_v1/data'
sys.path.insert(0, str(ROOT / 'src_state512_v25'))
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PAPER / 'scripts'))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from model import BlocksRuleWorld
from stripsnav.io import load_committed
from paper_blocks_models import make_model
from predictive_nesy.environment import FrozenVision
from predictive_nesy.selected_row_objectives import demonstration_objective
from predictive_nesy.graph_subgoal_interpreter import export_program, plan
from predictive_nesy.gate import NeuralBoundaryGate


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def status(out, **kw):
    row = dict(time=time.time(), pid=os.getpid(), **kw)
    write(out / 'status.json', row)
    print(json.dumps(row), flush=True)


def save(path, model, optimizer=None, **meta):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=None if optimizer is None else optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), **meta), tmp)
    os.replace(tmp, path)
    write(path.with_suffix('.COMMITTED.json'), dict(path=str(path), sha256=sha(path), **meta))


class Images(Dataset):
    def __init__(self, names, size):
        self.names, self.size = names, size
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        with Image.open(DATA / self.names[i]) as im:
            a = np.array(im.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR), copy=True)
        x = torch.from_numpy(a).permute(2, 0, 1).float() / 255
        return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


@torch.no_grad()
def teacher_labels(names, out):
    ck = load_committed(OLD / 'world_model/best')
    network = make_model(ck['levels'], ck['n_actions'], ck.get('normalization', 'baseline')).cuda().eval()
    network.load_state_dict(ck['model'])
    labels = []
    for x in DataLoader(Images(names, 228), batch_size=64, num_workers=2, pin_memory=True):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            labels.append(network.encode(x.cuda())[2].cpu())
    ids = torch.cat(labels)
    symbols = ids.unique(sorted=True)
    assert 1 < len(symbols) <= 512
    # Random capacity-slot names avoid a geometry-based or ordered assignment.
    slots = torch.randperm(512, generator=torch.Generator().manual_seed(16103))[:len(symbols)]
    mapped = slots[torch.searchsorted(symbols, ids)]
    write(out / 'representation_teacher.json', dict(checkpoint=str(OLD / 'world_model/best'),
        commit=read(OLD / 'world_model/best/COMMITTED'), unique_anonymous_codes=len(symbols),
        opaque_code_to_slot=dict(zip(map(str, symbols.tolist()), slots.tolist())),
        inputs='training RGB only', reused_components='image encoder only; no old rules or action predictor',
        simulator_state_labels_used=False))
    del network, ck; torch.cuda.empty_cache()
    return mapped


@torch.no_grad()
def features(frontend, names, out=None):
    raw = []
    for i, x in enumerate(DataLoader(Images(names, 224), batch_size=32, num_workers=2, pin_memory=True)):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = F.adaptive_avg_pool2d(frontend.vision(x.cuda()), (7, 7)).flatten(2).transpose(1, 2)
        v = frontend.vocabulary
        z = F.normalize(((f.float() - v['mean']) / v['sd']) @ v['axes'], dim=-1)
        raw.append(z.cpu().half())
        if out is not None and i % 50 == 0:
            status(out, stage='cache_training_images', images=min(len(names), (i + 1) * 32), total=len(names))
    return torch.cat(raw)


@torch.no_grad()
def fit_visual_vocabulary(raw):
    # Same 64-category patch protocol; centers refitted only to training RGB.
    x = raw.flatten(0, 1).float().cuda()
    g = torch.Generator(device='cuda').manual_seed(16104)
    fit = x[torch.randperm(len(x), generator=g, device='cuda')[:65536]]
    centers = fit[torch.randperm(len(fit), generator=g, device='cuda')[:64]].clone()
    for _ in range(40):
        ids = torch.cdist(fit, centers).argmin(-1)
        sums = torch.zeros_like(centers).index_add_(0, ids, fit)
        counts = torch.bincount(ids, minlength=64)
        centers = torch.where(counts[:, None] > 0, sums / counts[:, None].clamp_min(1), centers)
    return centers


@torch.no_grad()
def quantize(raw, centers):
    return torch.cat([torch.cdist(x.float().cuda().flatten(0, 1), centers).argmin(-1).reshape(len(x), 49).cpu()
                      for x in raw.split(256)])


def prepared(model, codes, raw, before, goal):
    return model.prepare_pair(codes[before], raw[before], codes[goal], raw[goal])


def calibrate(model, codes, raw, labels, out, updates):
    model.set_stage('representation')
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for step in range(1, updates + 1):
        left = torch.randint(len(codes), (128,), device='cuda')
        right = torch.randint(len(codes), (128,), device='cuda')
        p = prepared(model, codes, raw, left, right)
        # Fixed-program definitions ground both current and another image.
        l = model.ground_features(p, codes[left]).squeeze(1)
        r = model.ground_features(p, codes[right]).squeeze(1)
        loss = (F.nll_loss(l.clamp_min(1e-9).log(), labels[left]) + F.nll_loss(r.clamp_min(1e-9).log(), labels[right])) / 2
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        opt.step()
        if step % 100 == 0 or step == 1:
            status(out, stage='representation_distillation', update=step, loss=float(loss.detach()),
                   sampled_accuracy=float(((l.argmax(-1) == labels[left]).float().mean() + (r.argmax(-1) == labels[right]).float().mean()) / 2))
    acc = 0; n = 0
    with torch.no_grad():
        for left in torch.arange(len(codes), device='cuda').split(64):
            right = (left * 37 + 11) % len(codes)
            p = prepared(model, codes, raw, left, right)
            pred = model.ground_features(p, codes[left]).argmax(-1).squeeze(1)
            acc += int((pred == labels[left]).sum()); n += len(left)
    report = dict(images=n, correct=acc, agreement=acc/n, training_only=True)
    write(out / 'representation_fit.json', report)
    save(out / 'representation.pt', model, opt, stage='representation', update=updates)
    return report


def make_windows(rows, name_index, labels):
    # Goal RGB is an observation later on a recorded legal trajectory. No true
    # stack states, shortest paths or preconditions are used to make targets.
    result = []
    for start in range(0, len(rows) - 9):
        for length in (1, 3, 5, 9):
            chunk = rows[start:start + length]
            if any(chunk[i]['next_image'] != chunk[i+1]['image'] for i in range(length-1)): continue
            frames = [name_index[chunk[0]['image']]] + [name_index[r['next_image']] for r in chunk]
            if labels[frames[0]] == labels[frames[-1]]: continue
            # Shift real action IDs by one; action zero only closes the plan.
            actions = [r['action'] + 1 for r in chunk] + [0]
            frames.append(frames[-1])
            h = len(actions)
            result.append(dict(frames=frames + [frames[-1]] * (11-len(frames)),
                               actions=actions + [0] * (10-h), h=h))
    return result


def train_rules(model, codes, raw, windows, out, updates):
    model.set_stage('rules')
    fixed = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    by_frames = torch.tensor([w['frames'] for w in windows], device='cuda')
    by_actions = torch.tensor([w['actions'] for w in windows], device='cuda')
    lengths = torch.tensor([w['h'] for w in windows], device='cuda')
    save(out / 'before_rules.pt', model, stage='rules', update=0)
    began = time.time()
    for step in range(1, updates+1):
        t = time.time(); idx = torch.randint(len(windows), (64,), device='cuda')
        frames, action, h = by_frames[idx], by_actions[idx], lengths[idx]
        p = prepared(model, codes, raw, frames[:, 0], frames[torch.arange(len(idx), device='cuda'), h])
        valid = torch.arange(10, device='cuda')[None] < h[:, None]
        data = dict(task=torch.zeros(len(idx), dtype=torch.long, device='cuda'), start=torch.zeros(len(idx), dtype=torch.long, device='cuda'),
                    targets=codes[frames], actions=action, action_valid=valid, supervised=valid, goal_index=h)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = demonstration_objective(model, data, p)
        if not torch.isfinite(loss): raise FloatingPointError(metrics)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        opt.step(); torch.cuda.synchronize()
        row = dict(update=step, seconds=time.time()-t, **metrics)
        with (out/'train.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if step % 100 == 0 or step == 1:
            status(out, stage='rules_training', **row, elapsed=time.time()-began,
                   memory_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if step % 500 == 0 or step == updates:
            save(out/'latest.pt', model, opt, stage='rules', update=step)
    assert all(torch.equal(p.detach().cpu(), fixed[n]) for n,p in model.named_parameters() if n in fixed)
    write(out/'frozen_representation_audit.json', dict(passed=True, tensors=len(fixed)))
    return model


def serializable_program(p):
    return dict(successors=p.transitions.argmax(-1).astype(np.int16).tolist(),
                confidence=p.transitions.max(-1).tolist(), subgoal=p.subgoal.tolist(), program_sha256=p.digest())


def scoring_transition():
    """Load the unchanged, pure scoring function without importing a renderer.

    paper_blocks_data.py imports PyBullet solely for RGB rendering. Its original
    ACTIONS constant and transition function are sufficient for the same exact
    execution scorer used by the previous benchmark. Never called in training.
    """
    source=PAPER/'scripts/paper_blocks_data.py'
    tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='transition') or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ACTIONS' for t in n.targets))]
    assert len(nodes)==2
    ns={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    return ns['transition']


@torch.no_grad()
def evaluate(checkpoint, codes, raw, queries, lookup, out):
    # True state access starts only here, after training checkpoint commitment.
    transition=scoring_transition()
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model = BlocksRuleWorld().cuda().eval(); model.load_state_dict(saved['model']); model.requires_grad_(False)
    gate = NeuralBoundaryGate([model]); results=[]; began=time.time()
    for begin in range(0, len(queries), 8):
        part = queries[begin:begin+8]
        starts = torch.tensor([lookup[q['start_image']] for q in part], device='cuda')
        goals = torch.tensor([lookup[q['goal_image']] for q in part], device='cuda')
        with gate.boundary(0):
            generated = model.generate_pair(codes[starts], raw[starts], codes[goals], raw[goals])
            first = model.ground(generated, codes[starts]).argmax(-1).cpu().numpy()
        for i, q in enumerate(part):
            # Only the frozen arrays, integer initial state and neural subgoal
            # reach the solver. The private true state and action validity do not.
            allowed={'definitions','transitions','subgoal_logits','start','context_raw'}
            program = export_program({k:v for k,v in generated.items() if k in allowed}, i)
            found = None
            # Blocks task asks to attain a goal arrangement, so choose the
            # shortest complete exact-goal proof, within the same ten-step cap.
            for horizon in range(1,11):
                candidate = plan(program, first[i], horizon=horizon)
                if candidate['found']: found=candidate; break
            actions = tuple(found['actions']) if found else ()
            # Audit every symbolic step independently before physical execution.
            symbolic = int(first[i,0]); proof=[]
            for a in actions:
                nxt = int(program.transitions[symbolic,a].argmax())
                proof.append([symbolic,a,nxt]); symbolic=nxt
            assert not found or symbolic==int(program.subgoal[0])
            actual = tuple(q['start']); actual_states=[list(actual)]; invalid=False
            for a in actions:
                if a==0: break
                nxt = transition(actual, a-1)
                if nxt is None: invalid=True; break
                actual=nxt; actual_states.append(list(actual))
            success=found is not None and not invalid and actual==tuple(q['goal'])
            record=dict(query_id=q['query_id'], shortest_horizon=q['shortest_horizon'], start_image=q['start_image'],goal_image=q['goal_image'],
                        initial_symbol=first[i].tolist(), generated_goal=program.subgoal.tolist(), plan=found,
                        proof=proof, actual_states=actual_states, invalid_action=invalid, success=success,
                        neural_calls_during_execution=0, rules_sha256=program.digest())
            results.append(record)
            # Full arrays for a small fixed set, compact exact proofs for all.
            if len(results)<=4: program.save(out/f'program_{len(results):03d}.npz')
        if begin % 160 == 0: print(json.dumps(dict(stage='sealed_checkpoint_evaluation',checkpoint=checkpoint.name,completed=len(results),total=len(queries))),flush=True)
        del generated
    gate.close()
    groups={}
    for h in sorted({r['shortest_horizon'] for r in results}):
        subset=[r for r in results if r['shortest_horizon']==h]
        groups[str(h)]=dict(tasks=len(subset),successes=sum(r['success'] for r in subset),success_rate=sum(r['success'] for r in subset)/len(subset))
    summary=dict(tasks=len(results),successes=sum(r['success'] for r in results),success_rate=sum(r['success'] for r in results)/len(results),
                 by_horizon=groups,invalid_action_tasks=sum(r['invalid_action'] for r in results),
                 no_plan_tasks=sum(r['plan'] is None for r in results),boundary_violations=gate.violations,
                 all_executed_actions_from_complete_symbolic_plans=True,seconds=time.time()-began,checkpoint_sha256=sha(checkpoint))
    write(out/'summary.json',summary);write(out/'episodes.json',results)
    return summary


def evaluate_existing(out):
    """Resume evaluation of sealed weights; never run another optimizer step."""
    for name in ['before_rules.pt','latest.pt']:
        assert sha(out/name)==read((out/name).with_suffix('.COMMITTED.json'))['sha256']
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    v=torch.load(out/'vision_adapter.pt',map_location='cuda',weights_only=False)
    frontend=FrozenVision(v['vision_config'])
    query_path=OLD/'symbolic/strict_queries.json'
    assert sha(query_path)==read(out/'split_audit.json')['queries_sha256']
    queries=read(query_path);names=sorted({q[k] for q in queries for k in ['start_image','goal_image']})
    raw=features(frontend,names);codes=quantize(raw,v['centers']).cuda();raw=raw.cuda()
    results={}
    for name,checkpoint in [('before_rules',out/'before_rules.pt'),('after_rules',out/'latest.pt')]:
        folder=out/name;folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():
            results[name]=read(folder/'summary.json')
            assert results[name]['checkpoint_sha256']==sha(checkpoint)
        else:results[name]=evaluate(checkpoint,codes,raw,queries,{n:i for i,n in enumerate(names)},folder)
    write(out/'comparison.json',dict(representation_fit=read(out/'representation_fit.json'),results=results,
                                    manifest=read(out/'manifest.json'),scorer_source_sha256=sha(PAPER/'scripts/paper_blocks_data.py')))
    status(out,stage='complete',results=results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--representation-updates',type=int,default=2000);parser.add_argument('--updates',type=int,default=3000);parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert out.resolve().is_relative_to(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') in {'3','4','5','7'}
    if args.evaluate_only:return evaluate_existing(out)
    assert not (out/'manifest.json').exists(),'Do not overwrite a previous experiment'
    torch.manual_seed(16101);np.random.seed(16101);random.seed(16101);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    source=ROOT/'run_state512_v25/checkpoint_003000.pt';saved=torch.load(source,map_location='cpu',weights_only=False)
    vision_config=saved['manifest']['frozen_vision'];frontend=FrozenVision(vision_config)
    model=BlocksRuleWorld().cuda()
    state=model.state_dict();copied=[]
    # Transfer only the visual-state front end. Rule/goal networks learn anew.
    prefixes=('tokens.','observation.','context_read.','history.','context_norm.','definitions.')
    for n,v in saved['model'].items():
        if n.startswith(prefixes) and n in state and state[n].shape==v.shape:state[n]=v;copied.append(n)
    model.load_state_dict(state);del state,saved
    rows=read(DATA/'transitions.json');names=sorted({r[k] for r in rows for k in ['image','next_image']});lookup={n:i for i,n in enumerate(names)}
    manifest=dict(task='existing reconstructed BlocksWorld-3',train_transitions=len(rows),train_images=len(names),
        train_rows_sha256=sha(DATA/'transitions.json'),model='current State512RuleWorld with image-goal adapter',
        states=512,real_actions=18,stop_index=0,opaque_real_action_offset=1,old_rules_reused=False,
        no_maps=True,no_handwritten_predicates=True,no_simulator_state_supervision=True,
        image_encoder='frozen current ConvNeXt-Tiny; training-only patch vocabulary adaptation',
        representation_supervision='old learned image-only anonymous FSQ IDs; no physical state labels',
        image_goal_adaptation=True,history_length=1,complete_open_loop_plan=True,maximum_plan_actions=10,
        inference_goal_replacement=False,learning_rate=3e-4,training_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        representation_updates=args.representation_updates,rule_updates=args.updates,copied_state_tensors=copied,
        representation_frozen_during_rule_training=True,created=time.time())
    write(out/'manifest.json',manifest);status(out,stage='preparing')
    targets=teacher_labels(names,out).cuda()
    raw=features(frontend,names,out);centers=fit_visual_vocabulary(raw);codes=quantize(raw,centers).cuda();raw=raw.cuda()
    torch.save(dict(centers=centers.cpu(),vision_config=vision_config),out/'vision_adapter.pt')
    raw_hashes={sha(DATA/n) for n in names}
    fit=calibrate(model,codes,raw,targets,out,args.representation_updates)
    windows=make_windows(rows,lookup,targets.cpu().tolist());write(out/'training_windows.json',dict(count=len(windows),lengths=[1,3,5,9],initial_equals_goal_removed=True))
    train_rules(model,codes,raw,windows,out,args.updates)
    status(out,stage='training_complete_test_start',representation_fit=fit)
    # Evaluation manifest and its private labels are opened for the first time.
    query_path=OLD/'symbolic/strict_queries.json';queries=read(query_path)
    eval_names=sorted({q[k] for q in queries for k in ['start_image','goal_image']})
    overlap=[n for n in eval_names if sha(DATA/n) in raw_hashes]
    assert not overlap,'Test pixels overlap training'
    eval_raw=features(frontend,eval_names);eval_codes=quantize(eval_raw,centers).cuda();eval_raw=eval_raw.cuda()
    del model,raw,codes;torch.cuda.empty_cache()
    write(out/'split_audit.json',dict(training_images=len(names),evaluation_images=len(eval_names),
        evaluation_queries=len(queries),identical_images=0,queries_sha256=sha(query_path),
        qualification='New endpoint images; the small finite domain and its legal transition types were seen in training. Not unseen-layout or unseen-rule generalization.',
        test_read_only_after_training_commit=True))
    results={}
    for name,checkpoint in [('before_rules',out/'before_rules.pt'),('after_rules',out/'latest.pt')]:
        folder=out/name;folder.mkdir();results[name]=evaluate(checkpoint,eval_codes,eval_raw,queries,{n:i for i,n in enumerate(eval_names)},folder)
    write(out/'comparison.json',dict(representation_fit=fit,results=results,manifest=manifest))
    status(out,stage='complete',results=results)


if __name__=='__main__': main()
