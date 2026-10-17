"""Immutable checkpoint / training-only grounding audit, without a simulator."""
import json
import os
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from data import ROOT, Corpus, sha
from model import FactoredWorld, Config, hard


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    out=ROOT/'run_grounded_memory_v42';folder=out/'diagnostics';folder.mkdir(exist_ok=True)
    pin=folder/f'grounding_audit_{os.getpid()}.pt'
    for attempt in range(3):
        commit=json.loads((out/'latest.COMMITTED.json').read_text())
        try:os.link(out/commit['filename'],pin);break
        except FileNotFoundError:
            if attempt==2:raise
    try:
        assert sha(pin)==commit['sha256']
        saved=torch.load(pin,map_location='cpu',weights_only=False)
        model=FactoredWorld(Config(**saved['model_config'])).eval()
        model.load_state_dict(saved['model'],strict=True);c=model.cfg
        corpus=Corpus(0,1,ROOT/'onpolicy_successors_v40')
        data=corpus.sample(random.Random(421017),32,128,'cpu')
        x=data['branch_features'];prev=data['branch_previous'];t=data['branch_t'];start=data['branch_start']
        i=torch.arange(len(x))
        lo,z,_=model.observation_encoding(x);ob=hard(lo)
        lm,_,_=model.memory_sequence(ob,prev,z);mem=hard(lm)
        context=model.context_sequence(x,ob,prev,data['branch_task'])[i,start]
        future_logits,future_z,_=model.observation_encoding(data['successors']);future=hard(future_logits)
        a=torch.arange(c.actions)[None].expand(len(x),-1)
        m0=mem[i,t,None].expand(-1,c.actions,-1)
        o0=ob[i,t,None].expand(-1,c.actions,-1)
        evidence0=z[i,t,None].expand(-1,c.actions,-1)
        target=model.memory_step(m0,future,a,future_z)[0].argmax(-1)
        no_new_evidence=model.memory_step(m0,o0,a,evidence0)[0].argmax(-1)
        pm,po=model.transition(context,m0,o0,a)
        initial=m0.argmax(-1);changed=target!=initial
        seen=data['branch_onpolicy']
        def metrics(mask):
            labels=target[mask]; prediction=pm[mask].argmax(-1); mask_changed=changed[mask]
            return dict(outcomes=labels.numel(),memory_prediction_accuracy=float((prediction==labels).float().mean()),
                memory_persistence_baseline=float((initial[mask]==labels).float().mean()),
                changed_memory_targets=int(mask_changed.sum()),
                changed_memory_accuracy=float(((prediction==labels)&mask_changed).sum()/mask_changed.sum().clamp_min(1)),
                target_depends_on_actual_new_visual_evidence=float((labels!=no_new_evidence[mask]).float().mean()),
                distinct_target_memory_codes=int(labels.unique().numel()),
                observation_accuracy=float((po[mask].argmax(-1)==future_logits[mask].argmax(-1)).float().mean()))
        result=dict(time=time.time(),training_only=True,source_checkpoint=commit,
            actual_memory_grounding='Uses recorded successor images to generate targets; predictor sees current hard premises only',
            partitions=dict(all=metrics(torch.ones(len(x),dtype=torch.bool)),onpolicy=metrics(seen),original=metrics(~seen)),
            limitation='A fixed sampled training batch is a diagnostic, not navigation success or held-out prediction accuracy.')
        (folder/f'memory_grounding_{commit["update"]:06d}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result))
    finally:
        if pin.exists():pin.unlink()


if __name__=='__main__':main()
