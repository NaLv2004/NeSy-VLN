import collections,json,sys,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_recovery_v28';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,DataBank,atomic
cfg=read(root/'configs/recovery_mixture.json');setup(cfg);bank=DataBank(cfg)
paired=set(bank.paired_demo_ids);recovery=set(bank.recovery_demo_ids)
groups={'base':[eid for eid in bank.demo_ids if eid not in paired|recovery],
        'paired':bank.paired_demo_ids,'recovery':bank.recovery_demo_ids}
result={}
for name,ids in groups.items():
    rows=[r for eid in ids for r in bank.demos[eid] if r.get('supervision_mask',True)]
    counts=collections.Counter(r['action'] for r in rows)
    result[name]={'episodes':len(ids),'supervised_frames':len(rows),'counts':dict(counts),
        'straight_or_stop_fraction':sum(counts[a] for a in [0,4,11])/len(rows),
        'turning_fraction':sum(v for a,v in counts.items() if a not in [0,4,11])/len(rows),
        'context_only_frames':sum(not r.get('supervision_mask',True) for eid in ids for r in bank.demos[eid])}
record={'time':time.time(),'train_only':True,'sources':result,'not_navigation_success':True}
atomic(base/'training_action_distribution_v29.json',record);print(json.dumps(result,indent=2))
