"""A diagnostic decoder sees only saved Boolean symbols and task/version tokens.

Never deployed as a policy; no held-out development/fixed navigation tasks used.
"""
import json,os,random,sys,time,collections
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_robust_v7'
sys.path.insert(0,str(root));os.environ['CUDA_VISIBLE_DEVICES']='6'
import torch
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstrations_many

torch.set_num_threads(2);torch.manual_seed(1024);torch.cuda.set_per_process_memory_fraction(.16)
cfg=read(root/'configs/robust_multieffect.json');setup(cfg);bank=DataBank(cfg)
model=NavigationModel(cfg).to('cuda').eval()
checkpoint=root/'runs/robust_multieffect/evaluations/step_000030/checkpoint';load_checkpoint(checkpoint,model)
rng=random.Random(991);groups={}
for eid in bank.demo_ids:
    ep=bank.by_id[eid];groups.setdefault((ep['category'],ep['domain'],ep['distance_group']),[]).append(eid)
selected=[]
for key,ids in sorted(groups.items()):
    rng.shuffle(ids);selected.extend(ids[:16])
rows=[];start=time.time()
for begin in range(0,len(selected),8):
    policies=demonstrations_many(model,bank,selected[begin:begin+8])
    for p in policies:
        ep=bank.by_id[p.episode_id];category=int(ep['category']=='table')
        for c in p.chunks:
            for row in c['rows']:
                rows.append({'episode':p.episode_id,'category':category,'version':row['version'],
                    'codes':row['symbol_codes'],'perceptual_codes':row['perceptual_codes'],
                    'action':row['action'],'rule_action':max(range(17),key=lambda a:row['support'][a])})
    if begin%32==0:print(json.dumps({'encoded_episodes':min(begin+8,len(selected)),'frames':len(rows),'seconds':time.time()-start}),flush=True)
    del policies
peak=torch.cuda.max_memory_allocated()/2**30
del model;torch.cuda.empty_cache()
out=base/'diagnostics_20261011/discrete_information_probe';out.mkdir(exist_ok=True)
atomic(out/'symbol_rows.json',{'checkpoint':str(checkpoint),'rows':rows},compact=True)
test_ids=set()
for key,ids in sorted(groups.items()):test_ids.update(e for e in ids[:16][-4:])
train=torch.tensor([r['episode'] not in test_ids for r in rows]);test=~train
y=torch.tensor([r['action'] for r in rows]);rule=torch.tensor([r['rule_action'] for r in rows])
results={}
for source in ['codes','perceptual_codes']:
    # These are goal-category and revision-number tokens, not human-labelled predicates.
    x=torch.tensor([r[source]+[r['category'],1-r['category']]+[int(r['version']==i) for i in range(4)] for r in rows],dtype=torch.float)
    for kind in ['linear','mlp']:
        torch.manual_seed(339)
        decoder=torch.nn.Linear(x.shape[1],17) if kind=='linear' else torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.GELU(),torch.nn.Linear(128,128),torch.nn.GELU(),torch.nn.Linear(128,17))
        optimizer=torch.optim.AdamW(decoder.parameters(),lr=.005,weight_decay=.01)
        for update in range(300):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(decoder(x[train]),y[train]);loss.backward();optimizer.step()
        with torch.no_grad():
            prediction=decoder(x).argmax(-1)
            results[f'{source}_{kind}']={'training_accuracy':float((prediction[train]==y[train]).float().mean()),
                'held_training_trajectory_accuracy':float((prediction[test]==y[test]).float().mean()),
                'held_stop_recall':float((prediction[test&(y==0)]==0).float().mean())}
        torch.save(decoder.state_dict(),out/f'{source}_{kind}_diagnostic_only.pt')
    buckets={}
    for index,r in enumerate(rows):buckets.setdefault(tuple(x[index].tolist()),collections.Counter())[r['action']]+=1
    results[source+'_aliasing']={'distinct_keys':len(buckets),'empirical_best_possible_accuracy':sum(max(c.values()) for c in buckets.values())/len(rows),
        'frames_in_conflicting_keys':sum(sum(c.values()) for c in buckets.values() if len(c)>1)}
summary={'checkpoint':str(checkpoint),'frames':len(rows),'training_episodes':len(set(selected)-test_ids),'held_training_episodes':len(test_ids),
    'rule_training_accuracy':float((rule[train]==y[train]).float().mean()),'rule_held_training_trajectory_accuracy':float((rule[test]==y[test]).float().mean()),
    'majority_training_action_accuracy':int(torch.bincount(y[train]).max())/int(train.sum()),'results':results,
    'peak_diagnostic_gpu_gib':peak,'seconds':time.time()-start,'time':time.time(),
    'qualification':'Diagnostic decoding of frozen learned Boolean symbols, on existing training trajectories only. This is not a navigation success rate or an independent generalization benchmark. Decoders are not deployed.'}
atomic(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
