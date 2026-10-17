"""Recover causal boundary inputs by replay, never by another episode's image."""
import argparse, json, os, select, subprocess, sys, time, fcntl
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image
from predictive_nesy.environment import FrozenVision, make_bank
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.data import sha

ROOT = Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')

def atomic(path, obj):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj, indent=2)); os.replace(temp, path)

class Replay:
    def __init__(self, bank, out):
        sys.path.insert(0, '/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')
        from objectnav_rl.ipc import receive, send
        self.receive, self.send = receive, send
        self.log = (out / 'simulator.log').open('ab')
        env = dict(os.environ, MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        self.process = subprocess.Popen([bank.env['sim_python'], str(Path(__file__).with_name('replay_boundary_worker.py'))],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.call('init', experiment=bank.env, train_scenes=bank.provenance['train_layouts'])
    def call(self, op, **kwargs):
        self.send(self.process.stdin, dict(op=op, **kwargs))
        if not select.select([self.process.stdout], [], [], 180)[0]: raise TimeoutError('Keep existing replay handle; do not duplicate')
        r = self.receive(self.process.stdout)
        if not r['ok']: raise RuntimeError(r['error'])
        return r['result']
    def close(self):
        if self.process.poll() is None:
            self.call('close'); self.process.wait(timeout=15)
        self.log.close()

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--out', required=True); p.add_argument('--limit-per-bank', type=int, default=0)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1); args=p.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'); assert gpu in {'3','4','5','7'}
    assert 0 <= args.shard < args.shards
    torch.set_num_threads(2); out=Path(args.out).resolve(); assert out.is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True); logdir=out/f'shard_{args.shard}'; logdir.mkdir(exist_ok=True)
    lock=(logdir/'lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT/'run_definition_world_v12/manifest.json').read_text()); bank=make_bank()
    assert bank.provenance['train_layouts']==manifest['data']['train_layouts']
    jobs=[]
    for record in manifest['transition_banks']:
        folder=Path(record['path']); known=set(json.loads((folder/'manifest.json').read_text())['episodes'])
        paths=[f for f in sorted(folder.glob('*.npz')) if not f.name.endswith('.tmp.npz')]
        if args.limit_per_bank: paths=paths[:args.limit_per_bank]
        for path in paths:
            assert path.stem in known and bank.by_id[path.stem]['scene'] in bank.provenance['train_layouts']
            jobs.append(path)
    jobs=jobs[args.shard::args.shards]
    vision=FrozenVision(manifest['frozen_vision']); gate=NeuralBoundaryGate([vision.vision]); encoder=RawBoundaryGraphEncoder(vision,gate)
    sim=Replay(bank,logdir); totals=Counter(); reports=[]; started=time.time()
    try:
        for path in jobs:
            target=out/path.parent.name/(path.stem+'.npz'); target.parent.mkdir(exist_ok=True)
            if target.exists(): raise RuntimeError('Existing result must be explicitly audited before resume')
            with np.load(path,allow_pickle=False) as data:
                codes=data['codes'].copy(); actions=data['actions'].copy(); valid=data['probe_valid'].copy()
            frames=sim.call('replay',episode=bank.by_id[path.stem],actions=actions.tolist())['rgb']
            assert len(frames)==len(codes)
            # Offline training-data extraction, not a running navigation policy.
            with gate.boundary(0):
                tokens,features=encoder.encode_joint([Image.fromarray(rgb) for rgb in frames])
            tokens=tokens.cpu().numpy().astype(np.uint8); features=features.cpu().numpy()
            exact_frames=(tokens==codes).all(-1); available=np.zeros(len(codes),bool)
            for start in range(0,len(codes),10):
                exact=bool(exact_frames[max(0,start-10):start+1].all())
                available[start:min(start+10,len(codes))]=exact
            tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,features=features,recorded_codes=codes,replayed_codes=tokens,
                recorded_actions=actions,context_available=available)
            os.replace(tmp,target)
            r=dict(bank=path.parent.name,episode=path.stem,source_sha256=sha(path),output_sha256=sha(target),
                frames=len(codes),exact_frames=int(exact_frames.sum()),patch_matches=int((tokens==codes).sum()),patches=codes.size,
                states=int(valid.sum()),available_states=int((available&valid).sum()))
            reports.append(r)
            for k in ['frames','exact_frames','patch_matches','patches','states','available_states']:totals[k]+=r[k]
            status=dict(state='collecting',pid=os.getpid(),gpu=int(gpu),completed=len(reports),total=len(jobs),time=time.time(),seconds=time.time()-started,totals=dict(totals))
            atomic(logdir/'status.json',status)
            if len(reports)%5==0: print(json.dumps(status),flush=True)
        report=dict(training_only=True,source_manifest_sha256=sha(ROOT/'run_definition_world_v12/manifest.json'),
            train_layouts=bank.provenance['train_layouts'],frozen_vision=manifest['frozen_vision'],totals=dict(totals),reports=reports,
            raw_context_policy='Available only when every causal replayed patch code equals the immutable original history',
            actions='Immutable recorded actions; no new policy sampling or online corrections',seconds=time.time()-started)
        atomic(logdir/'report.json',report); atomic(logdir/'status.json',dict(status,state='complete',time=time.time()))
        print(json.dumps(dict(complete=True,totals=dict(totals),seconds=time.time()-started)),flush=True)
    finally:
        gate.close();sim.close()

if __name__=='__main__': main()
