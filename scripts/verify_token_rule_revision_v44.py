"""Real-data integration of low-frequency revision; no simulator or images."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch

base=Path('/home/user/xjy/nesy_vla_20261010');source=base/'spatial_automaton_v43'
sys.path.insert(0,str(base/'symbolic_paired_v12'))
from nesy_vla.runtime import read,atomic,sha
from nesy_vla.token_rule_revision import TokenFacts,TokenProgramReviser,convert_v43_trees,replay_tree_program

torch.set_num_threads(2);torch.manual_seed(4400)
commit=read(source/'COMMITTED');assert sha(source/'programs.json')==commit['program_sha256']
manifest=read(source/'manifest.json');rows=read(source/'training_rows.json');codes=np.load(source/'visual_tokens.npy')
schema=(64,)*49+(65,)*50;programs={c:convert_v43_trees(p,schema) for c,p in read(source/'programs.json').items()}
tasks=sorted(programs);examples=[];starts={}
for i,row in enumerate(rows):
    starts.setdefault(row['episode_id'],i)
    start=starts[row['episode_id']];step=i-start
    if step not in [0,10] or not row['supervised']:continue
    if sum(e['category']==row['category'] and e['step']==step for e in examples)>=4:continue
    states=[]
    for j in range(max(start,i-10),i+1):
        previous=codes[j-1] if j>start else np.full(49,64,dtype=np.int16)
        past_action=rows[j-1]['label'] if j>start else 64
        states.append(np.r_[codes[j],previous,past_action])
    examples.append({'category':row['category'],'step':step,'id':row['episode_id'],'label':row['label'],
                     'history':TokenFacts(torch.tensor(np.array(states),dtype=torch.long),schema)})
assert len(examples)==16
reviser=TokenProgramReviser(schema,len(tasks),len(programs[tasks[0]]))
optimizer=torch.optim.AdamW(reviser.parameters(),lr=1e-3)
max_error=0;initial=[]
for e in examples:
    history=e['history'];revision=reviser(tasks.index(e['category']),history,e['step'])
    facts=TokenFacts(history.codes[-1],schema)
    p,certificate,conclusions=replay_tree_program(programs[e['category']],facts,revision,e['step'])
    error=float((p-conclusions.mean(0)).abs().max());max_error=max(max_error,error)
    assert error<1e-6
    initial.append(float(-p[e['label']].clamp_min(1e-9).log().detach()))
gradients=[]
for iteration in range(4):
    losses=[]
    for e in examples:
        history=e['history'];revision=reviser(tasks.index(e['category']),history,e['step'])
        p,_,_=replay_tree_program(programs[e['category']],TokenFacts(history.codes[-1],schema),revision,e['step'])
        losses.append(-p[e['label']].clamp_min(1e-9).log())
    optimizer.zero_grad();loss=torch.stack(losses).mean();loss.backward()
    norms={name:sum(float(p.grad.square().sum()) for n,p in reviser.named_parameters() if n.startswith(name) and p.grad is not None)**.5
           for name in ['symbols','history','program']}
    assert norms['program']>0
    if iteration:assert norms['symbols']>0 and norms['history']>0
    gradients.append(norms);optimizer.step()
effect=[]
for e in examples:
    revision=reviser(tasks.index(e['category']),e['history'],e['step'])
    changed=e['history'].codes.clone();changed[:,:49]=(changed[:,:49]+1)%64
    alternate=reviser(tasks.index(e['category']),TokenFacts(changed,schema),e['step'])
    effect.append(float((revision.weights-alternate.weights).abs().sum().detach()))
assert max(effect)>1e-6
result={'time':time.time(),'source_commit':commit,'examples':len(examples),'revision_steps':[0,10],
        'initial_equal_weight_replay_max_error':max_error,'gradient_norms':gradients,
        'symbol_context_change_program_l1':effect,'parameter_count':sum(p.numel() for p in reviser.parameters()),
        'images_loaded':False,'scenes_or_geometric_states_loaded':False,'fixed_or_development_used':False,
        'qualification':'Four optimizer steps on16training examples verify integration/causality only; not navigation or generalization evidence.',
        'initial_nll':sum(initial)/len(initial),'last_training_loss':float(loss.detach())}
atomic(base/'token_rule_revision_integration_v44.json',result)
print(json.dumps(result,indent=2))
