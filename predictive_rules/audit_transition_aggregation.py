"""Audit completed transition banks without reading reward or semantic fields."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from predictive_nesy.subgoal_data import TransitionBank

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-manifest',required=True)
    p.add_argument('--bank',action='append',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();m=json.loads(Path(a.checkpoint_manifest).read_text());rows=[]
    for path in a.bank:
        bank=TransitionBank(path,m['data']['train_layouts']);stats=bank.refresh();reasons=collections.Counter()
        continuity=collections.Counter();schema=collections.Counter();branch_states=0;hist_lengths=collections.Counter()
        for name,d in bank.episodes.items():
            n=len(d['actions']);assert d['codes'].shape==(n,49) and d['successors'].shape==(n,17,49)
            assert d['probe_valid'].shape==(n,) and int(d['task']) in [0,1]
            assert np.all((d['codes']>=0)&(d['codes']<64)) and np.all((d['successors']>=0)&(d['successors']<64))
            assert np.all((d['actions']>=0)&(d['actions']<17))
            branch_states+=int(d['probe_valid'].sum());hist_lengths[n]+=1
            # Test only valid probes. Executed successor images are not assumed
            # identical when a probe was marked unreliable by its collector.
            for t in range(n-1):
                if d['probe_valid'][t]:
                    equal=np.array_equal(d['successors'][t,d['actions'][t]],d['codes'][t+1])
                    continuity['equal' if equal else 'different']+=1
                    if not equal:reasons[name]+=1
            with np.load(Path(path)/(name+'.npz'),allow_pickle=False) as raw:
                schema[','.join(sorted(raw.files))]+=1
        rows.append(dict(path=path,**stats,valid_states=branch_states,temporal_code_continuity=dict(continuity),
            different_examples=dict(reasons.most_common(8)),schema_counts=dict(schema),
            trajectory_length_counts=dict(hist_lengths),manifest=bank.manifest))
    result=dict(training_only=True,read_fields=['codes','successors','actions','probe_valid','task'],banks=rows)
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({**result,'banks':[{k:v for k,v in r.items() if k not in ['manifest','schema_counts']} for r in rows]}))
