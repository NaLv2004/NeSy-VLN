"""Fixed real-data backward comparison on CPU, no rollout or optimizer step."""
import json
from pathlib import Path
import random
import torch
from model import FactoredWorld
from data import ROOT,Corpus

torch.set_num_threads(3)
out=ROOT/'run_induced_effects_v39'
commit=json.loads((out/'latest.COMMITTED.json').read_text())
payload=torch.load(out/commit['filename'],map_location='cpu',weights_only=False)
corpus=Corpus(0,2)
long_names=[name for name in corpus.names if len(corpus.episodes[name])>=20]
assert long_names
corpus.names=long_names
data=corpus.sample(random.Random(39019),64,32,'cpu')
results=[]
for mode in ('full_straight_through','detached_memory_carrier','detached_history_carrier'):
    model=FactoredWorld();model.load_state_dict(payload['model'])
    model.detach_copy_evidence=False;model.detach_memory_carrier=False
    if mode=='detached_memory_carrier':
        original=model.memory_step
        def step(memory,observation,previous,initial=False):
            return original(None if memory is None else memory.detach(),observation,previous,initial)
        model.memory_step=step
    if mode=='detached_history_carrier':
        # Same forward values, cutting only backward recurrence within the
        # encoder's observed history. Predicted dynamics gradients are intact.
        from model import hard
        def sequence(observations,previous,hidden=None):
            carrier=hidden[0] if hidden is not None else None
            logits=[];states=[]
            for t in range(observations.shape[1]):
                lm,state=model.memory_step(carrier,observations[:,t],previous[:,t],initial=carrier is None)
                logits.append(lm);states.append(state);carrier=hard(lm).detach()
            return torch.stack(logits,1),torch.stack(states,1),carrier[None]
        model.memory_sequence=sequence
    torch.manual_seed(39019)
    loss,metrics=model(data);loss.backward()
    norms={name:float(p.grad.double().norm()) for name,p in model.named_parameters() if p.grad is not None}
    total=sum(v*v for v in norms.values())**.5
    results.append(dict(mode=mode,loss=float(loss.detach()),gradient_norm=total,
        top_parameters=sorted(norms.items(),key=lambda p:-p[1])[:8]))
history=[json.loads(line) for line in (out/'rank0.jsonl').read_text().splitlines() if line.strip()]
history=[row for row in history if row['update']<=300]
import numpy as np
values=np.asarray([row['gradient_norm'] for row in history])
result=dict(source_commit=commit,results=results,no_optimizer_step=True,no_rollouts=True,
    diagnostic_long_episode_pool=len(long_names),sampled_episodes=len(data['features']),
    first300_gradient_norm=dict(count=len(values),median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(values.max()),greater_than_1000=int((values>1000).sum())))
(out/'gradient_audit_long.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
