"""Training-scene feature throughput/numerical check; no policy training/eval."""
import copy,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from collect_closure import Probe,Policy
from model import FactoredWorld,Config
from batch_future import BatchFutureEncoder
from data import ROOT
from evaluate import atomic
from predictive_nesy.environment import FrozenVision,make_bank
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
OUT=ROOT/'counterfactual_closure_v63/throughput';OUT.mkdir(exist_ok=True)
saved=torch.load(ROOT/'counterfactual_closure_v63/source.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model']);frontend=FrozenVision(saved['manifest']['frozen_vision'])
gate=NeuralBoundaryGate([model,frontend.vision],interval=1);canonical=RawBoundaryGraphEncoder(frontend,gate)
bank=make_bank();sim=Probe(bank,OUT);reports=[]
tasks=[next(e for e in bank.train if e['domain']==domain and e['distance_group']==distance)
       for domain,distance in (('normal','near'),('normal','far'),('dense','medium'))]
# Dataset domain names differ across corpora; select by actual values if needed.
start=time.time()
try:
    for episode in tasks:
        initial=sim.call('reset',episode=episode)['rgb']
        policy=Policy(model,episode['category'],canonical,gate);action,_=policy.choose(initial)
        state=policy.steps[-1] if action is not None else policy.failed_proof
        second=np.stack([np.random.default_rng(631017+a).choice(17,2,replace=False) for a in range(17)])
        t=time.time();r=sim.call('probe_only',action=None,second_actions=second);probe_seconds=time.time()-t
        ims=[Image.fromarray(x) for x in np.concatenate([r['successors'],r['second_successors'].reshape(34,224,224,3)])]
        def labels(raw):
            lo,z,_=model.observation_encoding(raw[None],policy.context);lo=lo[0];z=z[0];o=lo.argmax(-1)
            def memory(m,o,a,z):
                logits,_=model.memory_step(F.one_hot(m,32).float(),F.one_hot(o,64).float(),a,z)
                return logits
            lm1=memory(torch.full((17,),state['memory'],device='cuda'),o[:17],torch.arange(17,device='cuda'),z[:17])
            lm2=memory(lm1.argmax(-1)[:,None].expand(-1,2),o[17:].reshape(17,2),
                torch.tensor(second,device='cuda'),z[17:].reshape(17,2,128))
            return lo,torch.cat([lm1,lm2.reshape(34,32)])
        with gate.boundary(0):
            torch.cuda.synchronize();t=time.time();_,base=canonical.encode_joint(ims);torch.cuda.synchronize();base_time=time.time()-t
            base_labels=labels(base);row=dict(episode=episode['episode_id'],probe_seconds=probe_seconds,canonical_seconds=base_time,candidates=[])
            for size in (17,51,102):
                fast=BatchFutureEncoder(frontend,gate,size);fast.encode(ims);torch.cuda.synchronize()
                measurements=[]
                for _ in range(3):
                    t=time.time();raw=fast.encode(ims);torch.cuda.synchronize();measurements.append(time.time()-t)
                fast_labels=labels(raw)
                row['candidates'].append(dict(batch=size,seconds=float(np.median(measurements)),
                    feature_max_error=float((base-raw).abs().max()),
                    observation_id_matches=int((base_labels[0].argmax(-1)==fast_labels[0].argmax(-1)).sum()),
                    memory_id_matches=int((base_labels[1].argmax(-1)==fast_labels[1].argmax(-1)).sum()),
                    observation_logit_max_error=float((base_labels[0]-fast_labels[0]).abs().max()),
                    memory_logit_max_error=float((base_labels[1]-fast_labels[1]).abs().max()),images=len(ims),
                    peak_gib=torch.cuda.max_memory_allocated()/2**30))
        reports.append(row)
    atomic(OUT/'batch_benchmark.json',dict(time=time.time(),seconds=time.time()-start,rows=reports,
        scope='Three training-scene first-state probe sets; no validation tasks, no model updates, no navigation-performance estimate'))
    print(json.dumps(reports),flush=True)
finally:sim.close();gate.close()
