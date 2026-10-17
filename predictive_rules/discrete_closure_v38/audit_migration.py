"""CPU-only real-corpus audit, with no navigation rollouts or updates."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import torch
from data import Corpus,ROOT,sha
from model import FactoredWorld,hard

torch.set_num_threads(3)
old_source=ROOT/'src_learned_persistence_v37/model.py'
spec=importlib.util.spec_from_file_location('old_model_v37',old_source)
old_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old_module;spec.loader.exec_module(old_module)
old_run=ROOT/'run_learned_persistence_v37'
commit=json.loads((old_run/'best.COMMITTED.json').read_text())
path=old_run/commit['filename'];assert sha(path)==commit['sha256']
checkpoint=torch.load(path,map_location='cpu',weights_only=False)
old=old_module.FactoredWorld();old.load_state_dict(checkpoint['model']);old.eval()
new=FactoredWorld();new.load_v37(checkpoint['model']);new.eval()
corpus=Corpus(0,2);data=corpus.sample(random.Random(38001),64,16,'cpu')
with torch.no_grad():
    x=data['features'];o=hard(old.observe(x));prev=data['previous']
    assert torch.equal(old.observe(x),new.observe(x))
    assert torch.equal(old.context_sequence(x,o,prev,data['task']),new.context_sequence(x,o,prev,data['task']))
    ml,hidden,_=old.memory_sequence(o,prev);m=hard(ml)
    valid=torch.arange(x.shape[1]-1)[None]<data['end'][:,None]
    # Even with true successor observations supplied, replacing hidden history
    # by the currently available categorical memory changes old labels.
    estimate,_=new.memory_step(m[:,:-1],o[:,1:],prev[:,1:])
    mismatch=((estimate.argmax(-1)!=ml[:,1:].argmax(-1))&valid).sum()
    full,_,_=new.memory_sequence(o,prev)
    j=data['start'];i=torch.arange(len(x))
    restart,_,_=new.memory_sequence(o[i,j+1,None],prev[i,j+1,None],hard(full[i,j])[None])
    assert torch.equal(restart[:,0],full[i,j+1])
    # Inspect hidden-state dependence without making a claim about causality
    # of task failure; the categorical reconstruction itself is lossy.
    result=dict(time=time.time(),source=str(path),source_commit=commit,episodes=len(x),
        transitions=int(valid.sum()),old_label_changes_after_categorical_restart=int(mismatch),
        old_label_change_rate=float(mismatch/valid.sum()),
        preserved_frame_encoder_and_context=True,new_exact_discrete_restart=True,
        no_new_rollouts=True,no_optimizer_step=True)
loss,metrics=new(data);loss.backward()
assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
result['real_data_loss']=float(loss.detach());result['real_data_all_gradients_finite']=True
target=ROOT/'src_discrete_closure_v38/migration_audit.json'
target.write_text(json.dumps(result,indent=2));print(json.dumps(result))
