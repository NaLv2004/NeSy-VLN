"""Read-only correspondence audit of immutable training banks; no model calls."""
import json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'diagnostics/raw_boundary_coverage_v19.json';assert not out.exists()
manifest=json.loads((root/'run_definition_world_v12/manifest.json').read_text())
corpus=Path(manifest['data']['source']);rows=json.loads((corpus/'training_rows.json').read_text());tokens=np.load(corpus/'visual_tokens.npy')
groups=defaultdict(list)
for i,r in enumerate(rows):groups[r['episode_id']].append(i)
reports=[];total=Counter();missing_contexts=[]
for record in manifest['transition_banks']:
    folder=Path(record['path']);bm=json.loads((folder/'manifest.json').read_text());allowed=set(bm['episodes']);counts=Counter();starts=defaultdict(Counter)
    for path in sorted(folder.glob('*.npz')):
        if path.name.endswith('.tmp.npz'):continue
        assert path.stem in allowed
        with np.load(path,allow_pickle=False) as data:
            codes=data['codes'];valid=data['probe_valid'];group=groups.get(path.stem)
            for start in sorted({int(t)//10*10 for t in np.flatnonzero(valid)}):
                n=sum(bool(valid[t]) for t in range(start,min(start+10,len(valid))))
                begin=max(0,start-10);reason=None
                if group is None:reason='episode_not_in_demonstration_corpus'
                elif start>=len(group):reason='demonstration_ended_before_revision'
                elif not np.array_equal(tokens[group[begin:start+1]],codes[begin:start+1]):reason='actual_recorded_history_differs'
                key='available' if reason is None else reason
                counts[key]+=n;starts[start][key]+=n;counts['states']+=n
                if reason is not None:missing_contexts.append(dict(bank=str(folder),episode=path.stem,start=start,reason=reason,states=n))
    total.update(counts);reports.append(dict(path=str(folder),counts=dict(counts),coverage=counts['available']/max(counts['states'],1),
        by_start={str(k):dict(v) for k,v in starts.items()}))
result=dict(training_only=True,no_new_training_or_navigation=True,total=dict(total),coverage=total['available']/max(total['states'],1),
    banks=reports,missing_unique_contexts=len(missing_contexts),missing_contexts=missing_contexts,
    interpretation='Weighted coverage for the existing uniform-state transition sampler; missing histories are deliberately never paired to a different demonstration')
out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='missing_contexts'}),flush=True)
