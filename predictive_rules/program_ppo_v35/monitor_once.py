"""Read saved training evidence only; never launch additional evaluation."""
import json
from pathlib import Path
import statistics
import subprocess
import time
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_program_ppo_v35'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def rows(p):
    result=[]
    if p.exists():
        for line in p.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass
    return result
now=time.time();marker=ROOT/'program_ppo_v35_monitor_latest.json'
old=read(marker) or read(ROOT/'program_ppo_v35_startup.json')
if old and now<old['observed_at']+600:
    print(json.dumps(dict(deferred=True,current_time=now,next_check_not_before=old['observed_at']+600)))
    raise SystemExit
training=rows(OUT/'train.jsonl');validation=rows(OUT/'validation.jsonl')
supervisor=read(OUT/'supervisor.json') or {};pids=[supervisor.get('pid')]+supervisor.get('worker_pids',[])
processes={}
for pid in pids:
    if pid is None:continue
    path=Path('/proc',str(pid),'cmdline')
    processes[str(pid)]=path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else None
sample=training[-10:]
result=dict(observed_at=now,supervisor=supervisor,processes=processes,status=read(OUT/'status.json'),
    ranks={str(i):dict(status=read(OUT/f'rank{i}/status.json'),error=read(OUT/f'rank{i}/error.json')) for i in range(2)},
    latest_commit=read(OUT/'latest.COMMITTED.json'),best_commit=read(OUT/'best.COMMITTED.json'),
    completed_updates=len(training),last_updates=training[-3:],validation=validation,
    recent_training=dict(updates=len(sample),successes=sum(r['successes'] for r in sample),
        episodes=sum(r['episodes'] for r in sample),
        success_rate=sum(r['successes'] for r in sample)/max(1,sum(r['episodes'] for r in sample)),
        mean_seconds=statistics.mean(r['seconds'] for r in sample) if sample else None),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
marker.write_text(json.dumps(result,indent=2));print(json.dumps(result))
