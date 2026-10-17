"""Empirical conditional ambiguity, on immutable training-only probe cases."""
from collections import defaultdict,Counter
import json,math,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'diagnostic_sparse_support_v44'
with np.load(out/'fixed_training_cases.npz') as data:
    memory=data['memory'].argmax(-1).reshape(-1)
    observation=data['observation'].argmax(-1).reshape(-1)
    label_o=data['target_o'].reshape(-1);label_m=data['target_m'].reshape(-1)
    heldout=np.repeat(data['heldout'],17);action=np.tile(np.arange(17),len(data['heldout']))
    context=np.repeat(data['context'],17,axis=0)
    positions=np.repeat(data['t'],17)
fit=np.flatnonzero(~heldout);test=np.flatnonzero(heldout)
results={}
for target,labels in [('next_observation',label_o),('next_memory',label_m)]:
    for inputs,keys in [('O_A',list(zip(observation,action))),('M_O_A',list(zip(memory,observation,action))),
        ('exact_C_M_O_A',[(context[i].tobytes(),int(memory[i]),int(observation[i]),int(action[i])) for i in range(len(action))])]:
        table=defaultdict(Counter);examples=defaultdict(list)
        for i in fit:table[keys[i]][int(labels[i])]+=1;examples[keys[i]].append(i)
        seen=[i for i in test if keys[i] in table]
        duplicated=[counter for counter in table.values() if sum(counter.values())>=2]
        majority_correct=sum(max(counter.values()) for counter in table.values())
        entropy=sum(sum(counter.values())*(-sum((n/sum(counter.values()))*math.log2(n/sum(counter.values())) for n in counter.values())) for counter in table.values())/len(fit)
        learned=[max(table[keys[i]],key=table[keys[i]].get) for i in seen]
        nn=[]
        for i in seen:
            candidates=examples[keys[i]]
            distance=((context[candidates]-context[i])**2).sum(-1)
            nn.append(labels[candidates[int(distance.argmin())]])
        results[target+'|'+inputs]=dict(fit_outcomes=len(fit),unique_premises=len(table),
            empirical_fit_accuracy_ceiling=majority_correct/len(fit),conditional_entropy_bits=entropy,
            repeated_premises=len(duplicated),conflicting_repeated_premises=sum(len(c)>1 for c in duplicated),
            heldout_outcomes=len(test),heldout_seen_premises=len(seen),heldout_coverage=len(seen)/len(test),
            heldout_accuracy_when_seen=float(np.mean(np.asarray(learned)==labels[seen])) if seen else None,
            nearest_boundary_context_accuracy_when_seen=float(np.mean(np.asarray(nn)==labels[seen])) if seen else None)
report=dict(time=time.time(),source=json.loads((out/'source.COMMITTED.json').read_text()),results=results,
    scope='Training-only cases with fixed encoders; heldout episodes excluded only from this count/decoder fit. No physical navigation, validation labels, or map.',
    caution='Empirical ceiling applies only to a context-free predictor on observed duplicate premises. Context-dependent generated tables may legitimately differ.')
(out/'code_aliasing.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
