"""Read-only corpus audit: temporal continuity, terminal balance and supervision masks."""
import collections,json
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010');source=root/'perceptual_sequences_v73/tail_1e4'
rows=json.loads((source/'training_rows.json').read_text());groups=collections.defaultdict(list)
for r in rows:groups[r['episode_id']].append(r)
lengths=collections.Counter();starts=collections.defaultdict(collections.Counter);bad=0
for eid,seq in groups.items():
    lengths[len(seq)]+=1
    for start in range(0,min(len(seq),32),10):
        sub=seq[start:start+min(10,32-start)];s=starts[start];s['windows']+=1;s['actions']+=len(sub)
        s['stop_in_window']+=any(r['label']==0 for r in sub);s['ends_within_two']+=len(sub)<=2
        s['all_supervised']+=all(r.get('supervised',True) for r in sub)
        s['masked_actions']+=sum(not r.get('supervised',True) for r in sub)
    for i,r in enumerate(seq[:-1]):bad+=r['label']==0
result=dict(rows=len(rows),episodes=len(groups),length_counts=dict(sorted(lengths.items())),windows_by_start=dict(starts),
    stop_before_final_row=bad,row_examples=[rows[i] for i in [0,len(rows)//2,len(rows)-1]])
out=root/'predictive_rules_20261015/subgoal_corpus_audit.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
