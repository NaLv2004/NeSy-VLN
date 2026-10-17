"""Real training corpus check, no simulator and no validation examples."""
import json
import random
from pathlib import Path
import torch
from data import Corpus,ROOT,sha
from model import FactoredWorld,Config

torch.set_num_threads(3)
old=ROOT/'run_grounded_memory_v42'
commit=json.loads((old/'best.COMMITTED.json').read_text())
path=old/commit['filename']
assert sha(path)==commit['sha256']
saved=torch.load(path,map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config']))
model.load_state_dict(saved['model'])
summary=[]
for rank in (0,1):
    corpus=Corpus(rank,2,str(ROOT/'onpolicy_successors_v40'))
    counts={label:list(map(len,pools)) for label,pools in corpus.phase_pools.items()}
    assert all(n>0 for pools in counts.values() for n in pools)
    data=corpus.sample(random.Random(431017+rank),32,32,'cpu')
    initial=(data['branch_t']==0).float().mean().item()
    early=((data['branch_t']>0)&(data['branch_t']<5)).float().mean().item()
    late=(data['branch_t']>=5).float().mean().item()
    assert (initial,early,late)==(.25,.25,.5)
    assert data['branch_onpolicy'].float().mean().item()==.5
    data['world_updates_encoder']=False
    model.zero_grad(set_to_none=True)
    loss,metrics=model(data);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    assert torch.isfinite(loss) and torch.isfinite(norm)
    for key in ('branch_initial_memory_accuracy','branch_initial_observation_accuracy',
                'branch_late_memory_accuracy','memory_persistence_baseline','changed_memory_accuracy'):
        assert torch.isfinite(metrics[key])
    summary.append(dict(rank=rank,pools=counts,phase_fractions=[initial,early,late],
        loss=float(loss.detach()),gradient_norm=float(norm),
        initial_memory_accuracy=float(metrics['branch_initial_memory_accuracy']),
        late_memory_accuracy=float(metrics['branch_late_memory_accuracy']),
        independent_training_only=True))
    del corpus,data
record=dict(passed=True,source_commit=commit,checks=summary,unchanged_model_and_planner=True,
    unchanged_data_split=True,meaning='Sampling and finite backward only, not improved navigation')
Path(__file__).with_name('sampling_contracts.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
