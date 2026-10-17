"""Compare already saved v39 code assignments on TRAIN images only.

No optimizer, new simulator rollout, validation image, or runtime intervention.
Hungarian alignment separates a vocabulary permutation from actual repartition.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from data import ROOT,sha

def linear_sum_assignment(cost):
    # Square Hungarian assignment; avoids adding a package to the live env.
    n=len(cost);u=np.zeros(n+1);v=np.zeros(n+1)
    p=np.zeros(n+1,dtype=int);way=np.zeros(n+1,dtype=int)
    for i in range(1,n+1):
        p[0]=i;j0=0;minimum=np.full(n+1,np.inf);used=np.zeros(n+1,dtype=bool)
        while True:
            used[j0]=True;i0=p[j0];delta=np.inf;j1=0
            for j in range(1,n+1):
                if used[j]:continue
                cur=cost[i0-1,j-1]-u[i0]-v[j]
                if cur<minimum[j]:minimum[j]=cur;way[j]=j0
                if minimum[j]<delta:delta=minimum[j];j1=j
            for j in range(n+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    rows=p[1:]-1;columns=np.arange(n)
    return rows,columns

check=-np.eye(64)[np.random.default_rng(3).permutation(64)]
ri,ci=linear_sum_assignment(check);assert check[ri,ci].sum()==-64

torch.set_num_threads(4)
spec=importlib.util.spec_from_file_location('v39_drift_model',ROOT/'src_induced_effects_v39/model.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
run=ROOT/'run_induced_effects_v39'
paths=[run/f'checkpoint_{i:06d}.pt' for i in (3000,5300,5336)]
models=[];commits=[]
for path in paths:
    commit=json.loads(path.with_suffix('.COMMITTED.json').read_text());assert sha(path)==commit['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=module.FactoredWorld(module.Config(**saved['model_config']));model.load_state_dict(saved['model']);model.eval()
    manifest=saved['manifest'];models.append(model);commits.append(commit);del saved
source=Path(manifest['data']['source'])
assert sha(source/'training_rows.json')==manifest['data']['rows_sha256']
rows=json.loads((source/'training_rows.json').read_text())
features=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
assignments=[[] for _ in models];margins=[[] for _ in models];began=time.time()
with torch.inference_mode():
    for begin in range(0,len(rows),256):
        x=torch.from_numpy(np.array(features[begin:begin+256],copy=True))
        for i,model in enumerate(models):
            p=model.observe(x).softmax(-1);top,ids=p.topk(2,dim=-1)
            assignments[i].extend(ids[:,0].tolist());margins[i].extend((top[:,0]-top[:,1]).tolist())
codes=np.asarray(assignments);supervised=np.asarray([r.get('supervised',True) for r in rows])
stop=np.asarray([int(r['label'])==0 for r in rows]);comparison=[]
for first,last in ((0,2),(1,2)):
    counts=np.zeros((64,64),np.int64);np.add.at(counts,(codes[first],codes[last]),1)
    before,after=linear_sum_assignment(-counts)
    equal=codes[first]==codes[last]
    comparison.append(dict(before_update=commits[first]['update'],after_update=commits[last]['update'],
        updates_elapsed=commits[last]['update']-commits[first]['update'],frames=len(rows),
        exact_id_retention=float(equal.mean()),
        best_permutation_retention=float(counts[before,after].sum()/len(rows)),
        supervised_id_retention=float(equal[supervised].mean()),
        terminal_id_retention=float(equal[stop&supervised].mean()),
        nonterminal_id_retention=float(equal[(~stop)&supervised].mean()),
        changed_frames=int((~equal).sum())))
result=dict(training_only=True,checkpoints=commits,comparisons=comparison,
    top1_top2_probability_margin=[dict(update=c['update'],mean=float(np.mean(m)),median=float(np.median(m)),fraction_below_point1=float((np.asarray(m)<.1).mean())) for c,m in zip(commits,margins)],
    seconds=time.time()-began,
    caveat='A changed ID may reflect useful representation learning; this comparison alone does not establish the cause of navigation failure.')
output=ROOT/'src_probabilistic_rules_v40/vocabulary_drift.json'
output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
