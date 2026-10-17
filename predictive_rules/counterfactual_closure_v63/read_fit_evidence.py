import json,time,subprocess
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
RUN=ROOT/'run_counterfactual_closure_v63'
read=lambda p:json.loads(p.read_text()) if p.exists() else {}
transition=[json.loads(s) for s in (RUN/'transitions.jsonl').read_text().splitlines()]
keys=('update','time','loss','first_o_accuracy','changed_accuracy','first_m_accuracy','conditional_o_accuracy','conditional_m_accuracy','open_o_accuracy','open_hard_m_accuracy')
ranks=[]
for rank in (0,1):
    rows=[json.loads(s) for s in (RUN/f'rank{rank}.jsonl').read_text().splitlines()]
    ranks.append(dict(rank=rank,updates=len(rows),first=rows[0],last=rows[-1],
        mean_update_seconds=sum(r['seconds'] for r in rows)/len(rows)))
status=[read(RUN/f'status_rank{i}.json') for i in (0,1)]
pids=[r['pid'] for r in status]+[read(RUN/'supervisor.json')['worker_pid']]
result=dict(time=time.time(),supervisor=read(RUN/'supervisor.json'),status=status,
    processes=[dict(pid=p,command=Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{p}/cmdline').exists() else '') for p in pids],
    transitions=[{k:r[k] for k in keys} for r in transition],ranks=ranks,
    best=read(RUN/'best.COMMITTED.json'),latest=read(RUN/'latest.COMMITTED.json'),
    manifest=read(RUN/'manifest.json'),summary=read(RUN/'dev_best_full80/summary.json'))
(RUN/'fit_observation.json').write_text(json.dumps(result,indent=2))
result['manifest']={k:result['manifest'].get(k) for k in ('architecture','batch_per_gpu','updates','learning_rate','checkpoint_selection','abstraction_and_goal_frozen')}
print(json.dumps(result))
