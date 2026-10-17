"""Offline anonymous-code diagnostics on existing TRAIN images only.

This fits frequency probes, never an executable policy, and changes no model.
Layout / teacher-action / terminal metadata are diagnostic labels only.
"""
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np
import torch
from data import ROOT,sha

torch.set_num_threads(4)
old=ROOT/'run_learned_persistence_v37'
commit=json.loads((old/'best.COMMITTED.json').read_text());path=old/commit['filename']
assert sha(path)==commit['sha256']
spec=importlib.util.spec_from_file_location('audit_model_v37',ROOT/'src_learned_persistence_v37/model.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
payload=torch.load(path,map_location='cpu',weights_only=False)
model=module.FactoredWorld();model.load_state_dict(payload['model']);model.eval()
source=Path(payload['manifest']['data']['source'])
assert sha(source/'training_rows.json')==payload['manifest']['data']['rows_sha256']
rows=json.loads((source/'training_rows.json').read_text())
features=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
codes=[]
with torch.no_grad():
    for start in range(0,len(features),256):
        x=torch.from_numpy(np.array(features[start:start+256],copy=True))
        codes.extend(model.observe(x).argmax(-1).tolist())
codes=np.asarray(codes);labels=np.asarray([int(r['label']) for r in rows])
tasks=np.asarray([['chair','table'].index(r['category']) for r in rows])
names=[r['episode_id'] for r in rows]
layouts=[re.search(r'P25_train_(\d+)_',r.get('source_episode_id',r['episode_id'])).group(1) for r in rows]
layout_ids=np.unique(layouts,return_inverse=True)[1]
supervised=np.asarray([r.get('supervised',True) for r in rows],bool)
def entropy(labels):
    count=np.bincount(np.asarray(labels,dtype=int));p=count[count>0]/count.sum()
    return float(-(p*np.log2(p)).sum())
def information(a,b):
    joint=a*(int(b.max())+1)+b
    mi=entropy(a)+entropy(b)-entropy(joint)
    return dict(mutual_information_bits=mi,label_entropy_bits=entropy(b),fraction_label_entropy=mi/max(entropy(b),1e-9))
def auc(y,p):
    pos=p[y];neg=p[~y]
    # 128-bin frequency scores; exact tied-rank Mann-Whitney statistic.
    return float(sum((x>neg).sum()+.5*(x==neg).sum() for x in pos)/max(1,len(pos)*len(neg)))
epnames=sorted(set(names));fold={name:i%5 for i,name in enumerate(epnames)}
held=np.asarray([fold[n]==0 for n in names]);fit=(~held)&supervised;test=held&supervised
key=codes*2+tasks;nkeys=model.cfg.observations*2
counts=np.zeros((nkeys,17),np.float64)
np.add.at(counts,(key[fit],labels[fit]),1)
prior=np.bincount(labels[fit],minlength=17)/fit.sum()
prob=(counts+.5*prior)/(counts.sum(-1,keepdims=True)+.5)
teacher_probe=float((prob[key[test]].argmax(-1)==labels[test]).mean())
stop_probability=prob[key[test],0];actual=labels[test]==0
predict=stop_probability>=.5
stop=dict(positives=int(actual.sum()),negatives=int((~actual).sum()),
    auc=auc(actual,stop_probability),
    balanced_accuracy=.5*(float(predict[actual].mean())+float((~predict[~actual]).mean())),
    recall=float(predict[actual].mean()),false_positive_rate=float(predict[~actual].mean()))
terminal=[]
for task in (0,1):
    positive=(labels==0)&(tasks==task)&supervised
    negative=(labels!=0)&(tasks==task)&supervised
    ids,count=np.unique(codes[positive],return_counts=True)
    shared=sum(int(count[j]) for j,c in enumerate(ids) if (codes[negative]==c).any())
    terminal.append(dict(category=['chair','table'][task],samples=int(positive.sum()),used_codes=len(ids),
        entropy_bits=entropy(codes[positive]),most_frequent_fraction=float(count.max()/count.sum()),
        fraction_terminal_samples_sharing_code_with_nonterminal=shared/max(1,int(positive.sum()))))
result=dict(time=time.time(),source=str(path),source_commit=commit,frames=len(rows),
    supervised_frames=int(supervised.sum()),episodes=len(epnames),
    code_information_about_layout=information(codes,layout_ids),
    code_information_about_teacher_action=information(codes[supervised],labels[supervised]),
    code_information_about_stop_label=information(codes[supervised],(labels[supervised]==0).astype(int)),
    terminal_code_distributions=terminal,frequency_probe=dict(fitting_frames=int(fit.sum()),testing_frames=int(test.sum()),
        teacher_action_accuracy=teacher_probe,majority_action_baseline=float((labels[test]==prior.argmax()).mean()),stop=stop),
    caveats=['All images were already in TRAIN. Frequency-probe episode folds are not new navigation validation.',
        'Probe tables are diagnostic only, never supplied to inference or symbolic planning.',
        'Shared code does not alone prove irreducible aliasing: the planner also has separate memory and generated context.'])
output=ROOT/'src_discrete_closure_v38/representation_audit.json'
output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
