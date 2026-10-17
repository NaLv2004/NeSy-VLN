"""Proof-policy training states and all-action visual successor supervision.

This is not evaluation and supplies no goal/action imitation labels. Immutable
weights choose each action solely by the same discrete interpreter as dev80.
Simulator probes and their features are stored after that choice, in a separate
data path. The policy receives only the actually executed result at next step.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import select
import subprocess
import sys
import time
import traceback
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from data import ROOT, sha
from model import FactoredWorld, Config, hard
from planner import Program, plan, audit
from evaluate import atomic
from predictive_nesy.environment import FrozenVision, make_bank
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate


class Probe:
    def __init__(self, bank, folder):
        sys.path.insert(0, '/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')
        from objectnav_rl.ipc import send, receive
        self.send, self.receive = send, receive
        self.log = (folder / 'simulator.log').open('ab')
        self.process = subprocess.Popen([bank.env['sim_python'], str(Path(__file__).with_name('onpolicy_worker.py'))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env=dict(os.environ, MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
        self.call('init', experiment=bank.env, train_scenes=bank.provenance['train_layouts'])

    def call(self, op, **kwargs):
        self.send(self.process.stdin, dict(op=op, **kwargs))
        if not select.select([self.process.stdout], [], [], 180)[0]:
            raise TimeoutError(f'Collector worker {self.process.pid}: {op}')
        response = self.receive(self.process.stdout)
        if not response['ok']:
            raise RuntimeError(response['error'])
        return response['result']

    def close(self):
        if self.process.poll() is None:
            self.call('close')
            self.process.wait(timeout=15)
        self.log.close()


class Policy:
    """Causal history, hard memory, rule renewal, and pure table derivation."""
    def __init__(self, model, task, encoder, gate):
        self.model, self.encoder, self.gate = model, encoder, gate
        self.c = model.cfg
        self.device = next(model.parameters()).device
        self.task = torch.tensor([['chair', 'table'].index(task)], device=self.device)
        self.features, self.observations, self.actions = [], [], []
        self.programs, self.steps = [], []
        self.program = None
        self.context = None
        self.grounding_allowed = False
        self.memory = self.predicted_observation = self.age = None

    @torch.no_grad()
    def choose(self, rgb):
        step = len(self.actions)
        with self.gate.boundary(step):
            _, raw = self.encoder.encode_joint([Image.fromarray(rgb)])
            observed_global = int(self.model.observe(raw).argmax(-1))
            observed = int(self.model.observe(raw,self.context).argmax(-1))
            self.features.append(raw[0])
            self.observations.append(observed_global)
            renew, probability = step == 0, 1.
            if step:
                memory = F.one_hot(torch.tensor([self.memory], device=self.device), self.c.memories).float()
                obs = F.one_hot(torch.tensor([observed], device=self.device), self.c.observations).float()
                predicted = F.one_hot(torch.tensor([self.predicted_observation], device=self.device), self.c.observations).float()
                probability = float(self.model.renewal_logits(memory, obs, predicted,
                    torch.tensor([self.age], device=self.device)).sigmoid())
                renew = self.age >= 5 and probability >= .5
            if renew:
                history = torch.stack(self.features)[None]
                previous = torch.tensor([[17] + self.actions], device=self.device)
                observations = F.one_hot(torch.tensor([self.observations], device=self.device), self.c.observations).float()
                context = self.model.context_sequence(history, observations, previous, self.task)[:, -1]
                self.context = context
                local_logits,evidence,_ = self.model.observation_encoding(history,context)
                observations = hard(local_logits)
                observed = int(local_logits[0,-1].argmax(-1))
                self.grounding_allowed = True
                try:
                    memory_logits, _, _ = self.model.memory_sequence(observations, previous,evidence)
                finally:
                    self.grounding_allowed = False
                self.memory = int(memory_logits[0, -1].argmax(-1))
                self.program = Program(**self.model.compile(context, self.task), observations=self.c.observations).validate()
                self.age = 0
                self.programs.append(dict(start=step, goal=self.program.goal, sha256=self.program.digest()))
        proof = plan(self.program, self.memory, observed, horizon=32-step)
        if not proof['found']:
            self.failed_proof = proof
            return None, raw[0].cpu().numpy()
        audit(self.program, proof)
        action = int(proof['actions'][0])
        self.steps.append(dict(step=step, observed=observed, memory=self.memory, action=action,
            renewal_probability=probability, renewed=renew, proof=proof))
        # The predicted hard memory is fixed BEFORE executing the action.
        self.memory = int(proof['next_memory'])
        self.predicted_observation = int(proof['predicted_observation'])
        self.age += 1
        return action, raw[0].cpu().numpy()

    def commit_execution(self, action):
        assert action == self.steps[-1]['action']
        self.actions.append(int(action))


def selected_tasks(bank, count, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for e in bank.train:
        groups[(e['domain'], e['category'], e['distance_group'])].append(e)
    assert count % len(groups) == 0
    quota = count // len(groups)
    tasks = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda e: e['episode_id'])
        assert len(values) >= quota
        tasks.extend(rng.sample(values, quota))
    rng.shuffle(tasks)
    return tasks


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tasks', type=int, default=120)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rank',type=int,required=True)
    args = parser.parse_args()
    assert args.rank in (0,1) and os.environ['CUDA_VISIBLE_DEVICES'] == str(6+args.rank) and 1 <= args.workers <= 4
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out).resolve()
    assert out.parent == ROOT and out.name == 'onpolicy_successors_v60'
    out.mkdir(exist_ok=True)
    lock = (out / f'collector_rank{args.rank}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = Path(args.checkpoint)
    commit = json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
    assert sha(checkpoint) == commit['sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    manifest = saved['manifest']
    model = FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    bank = make_bank()
    assert set(manifest['data']['train_layouts']) == set(bank.provenance['train_layouts'])
    tasks = selected_tasks(bank, args.tasks, 601017)
    assert not set(e['scene'] for e in tasks) & set(e['scene'] for e in bank.dev)
    provenance = dict(training_only=True, source_checkpoint=str(checkpoint), source_commit=commit,
        feature_shape=[49, 128], feature_dtype='float16',collection_round=3,root_without_proof_is_probed=True, policy='v57 best1000: all executed actions by unchanged exact categorical table derivation',
        data_fields=['features', 'successor_features', 'recorded_actions', 'probe_valid'],
        episode_ids=[e['episode_id'] for e in tasks], train_layouts=bank.provenance['train_layouts'],
        split_provenance=bank.provenance, max_actions=32, workers=args.workers,
        no_goal_or_imitation_supervision=True, privileged_state_labels=False,
        probes_available_to_policy=False, seed=601017,world_size=2)
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        assert old['source_commit']==commit and old['episode_ids']==provenance['episode_ids']
    if args.rank==0:atomic(out/'manifest.json', provenance)
    tasks=tasks[args.rank::2]
    data_dir = out/'counterfactual_onpolicy'
    data_dir.mkdir(exist_ok=True)
    completed = positions = executed = 0
    todo = []
    for e in tasks:
        path=data_dir/(e['episode_id']+'.npz')
        if path.exists() and path.with_suffix('.json').exists():
            record=json.loads(path.with_suffix('.json').read_text())
            assert record['output_sha256']==sha(path) and record['source_checkpoint_sha256']==commit['sha256']
            completed+=1;positions+=record['positions'];executed+=record['executed_actions']
        else:
            todo.append(e)
    frontend = FrozenVision(manifest['frozen_vision'])
    gate = NeuralBoundaryGate([model, frontend.vision], interval=1)
    encoder = RawBoundaryGraphEncoder(frontend, gate)
    def reject_inverse(*_):
        raise RuntimeError('Training-only inverse head cannot select collector actions')
    handle = model.inverse.register_forward_pre_hook(reject_inverse)
    active_policy=[None]
    def reject_memory_grounding(*_):
        assert active_policy[0] is not None and active_policy[0].grounding_allowed, 'Memory grounded outside renewal'
    memory_handle=model.memory.register_forward_pre_hook(reject_memory_grounding)
    sims=[];began=time.time()
    status=dict(state='starting',pid=os.getpid(),completed=completed,total=len(tasks),positions=positions,
        executed_actions=executed,time=time.time(),training_only=True)
    atomic(out/f'status_rank{args.rank}.json',status)
    try:
        for i in range(args.workers):
            folder=out/f'worker_{args.rank}_{i}';folder.mkdir(exist_ok=True)
            sims.append(Probe(bank,folder))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for offset in range(0,len(todo),args.workers):
                batch=todo[offset:offset+args.workers]
                responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(sims,batch)))
                rgb=[r['rgb'] for r in responses]
                policies=[Policy(model,e['category'],encoder,gate) for e in batch]
                current=[[] for _ in batch];future=[[] for _ in batch]
                ended=['budget' for _ in batch];active=list(range(len(batch)))
                for step in range(32):
                    requests={}
                    for i in active:
                        active_policy[0]=policies[i]
                        action,raw=policies[i].choose(rgb[i])
                        active_policy[0]=None
                        current[i].append(raw)
                        if action is None:
                            ended[i]='no_proof'
                            requests[i]=(action,pool.submit(sims[i].call,'probe_only'))
                        else:
                            requests[i]=(action,pool.submit(sims[i].call,'probe_and_execute',action=action))
                    next_active=[]
                    for i,(action,request) in requests.items():
                        response=request.result()
                        assert np.array_equal(response['current'],rgb[i]), 'Policy/probe current image differs'
                        # Only the frozen feature extractor sees counterfactual
                        # images. These arrays never enter a Policy instance.
                        with gate.boundary(step):
                            _,raw=encoder.encode_joint([Image.fromarray(x) for x in response['successors']])
                        future[i].append(raw.cpu().numpy())
                        if action is not None:
                            assert response['probe_matches_execution'],response['probe_rgb_error']
                            policies[i].commit_execution(action)
                            rgb[i]=response['actual']
                            if response['done']:
                                ended[i]='environment_done'
                            else:
                                next_active.append(i)
                    active=next_active
                    if not active:break
                for i,e in enumerate(batch):
                    path=data_dir/(e['episode_id']+'.npz')
                    temp=path.with_suffix('.tmp.npz')
                    policy=policies[i]
                    np.savez_compressed(temp,features=np.stack(current[i]).astype(np.float16),successor_features=np.stack(future[i]).astype(np.float16),
                        recorded_actions=np.asarray(policy.actions,np.int64),probe_valid=np.ones(len(current[i]),bool))
                    temp.replace(path)
                    record=dict(episode=e['episode_id'],scene=e['scene'],domain=e['domain'],category=e['category'],
                        distance_group=e['distance_group'],bank='counterfactual_onpolicy',training_only=True,
                        positions=len(current[i]),executed_actions=len(policy.actions),termination=ended[i],
                        output_sha256=sha(path),source_checkpoint_sha256=commit['sha256'],
                        programs=policy.programs,steps=policy.steps,failed_proof=getattr(policy,'failed_proof',None),
                        goal_supervision=False,probe_labels_available_to_policy=False)
                    atomic(path.with_suffix('.json'),record)
                    completed+=1;positions+=len(current[i]);executed+=len(policy.actions)
                status.update(state='collecting',completed=completed,positions=positions,executed_actions=executed,
                    time=time.time(),seconds=time.time()-began,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                atomic(out/f'status_rank{args.rank}.json',status)
                print(json.dumps(status),flush=True)
        atomic(out/f'status_rank{args.rank}.json',dict(status,state='complete',time=time.time()))
    except Exception as error:
        atomic(out/f'status_rank{args.rank}.json',dict(status,state='failed',time=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        memory_handle.remove();handle.remove();gate.close()
        for sim in sims:sim.close()


if __name__=='__main__':main()
