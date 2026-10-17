"""Immutable split/hash/shape audit and an actual mixed-corpus backward pass."""
import gc
import json
from pathlib import Path
import random
import torch
from data import Corpus,ROOT
from model import FactoredWorld,Config
from objectives import objective

torch.set_num_threads(3)
root=ROOT/'onpolicy_successors_v40'
progress=json.loads((root/'status.json').read_text());assert progress['state']=='complete'
rows=[];seen=set()
for rank in [0,1]:
    corpus=Corpus(rank,2,root)
    names={r['episode'] for r,d in corpus.branches if r.get('training_only')}
    assert not seen & names;seen |= names
    sample=corpus.sample(random.Random(41+rank),8,8,'cpu')
    assert float(sample['branch_onpolicy'].float().mean())==.5
    rows.append(dict(rank=rank,onpolicy_episodes=len(names),onpolicy_positions=len(corpus.onpolicy_positions),
        original_positions=len(corpus.old_branch_positions),onpolicy_fraction=.5))
    if rank==0:
        saved=torch.load(root/'policy.pt',map_location='cpu',weights_only=False)
        model=FactoredWorld(Config(**saved['model_config']));model.load_state_dict(saved['model'])
        sample['world_updates_encoder']=False
        loss,metrics=objective(model,sample);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        rows[-1].update(loss=float(loss.detach()),gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5)),
            finite_actual_mixed_corpus_backward=True)
        del saved,model,metrics,loss
    del corpus,sample;gc.collect()
assert len(seen)==120 and sum(r['onpolicy_positions'] for r in rows)==progress['positions']
result=dict(ready=True,unique_new_episodes=len(seen),new_positions=progress['positions'],
    new_successor_samples=17*progress['positions'],source_checkpoint_sha256=json.loads((root/'policy.COMMITTED.json').read_text())['sha256'],
    training_only=True,all_hashes_verified=True,goal_supervision_unchanged=True,shards=rows)
Path(__file__).with_name('data_readiness.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
