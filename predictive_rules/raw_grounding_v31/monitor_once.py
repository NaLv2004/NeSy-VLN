"""Read existing progress only; never launches a rollout or evaluation."""
import json,statistics,subprocess,time
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=r/'run_raw_grounding_v31'
previous=r/'raw_grounding_v31_monitor_latest.json'
minimum=1792163598.7714667
if previous.exists():minimum=max(minimum,json.loads(previous.read_text())['observed_at']+600)
now=time.time()
if now<minimum:
    print(json.dumps(dict(deferred=True,next_check_not_before=minimum,current_time=now)));raise SystemExit
def read(path):return json.loads(path.read_text()) if path.exists() else None
launch=read(r/'raw_grounding_v31_launcher.json');collector=read(r/'raw_successor_features_v31/parallel_render_launcher.json')
processes={}
for item in [launch,collector]:
    pid=item['pid'];path=Path('/proc',str(pid),'cmdline')
    processes[pid]=path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else None
rows=[]
with (out/'train.jsonl').open() as f:
    for line in f:
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
validation=[]
if (out/'validation.jsonl').exists():
    validation=[json.loads(line) for line in (out/'validation.jsonl').read_text().splitlines()]
windows={}
for name,values in [('first100',rows[:100]),('last100',rows[-100:])]:
    if values:
        keys=['transition_accuracy','counterfactual_accuracy','proof_action_accuracy','subgoal_exact_accuracy','online_target_state_agreement','branch_inverse_accuracy','seconds']
        windows[name]=dict(first_update=values[0]['update'],last_update=values[-1]['update'],count=len(values),
            metrics={key:statistics.mean(v[key] for v in values if key in v) for key in keys})
collection=r/'raw_successor_features_v31'
result=dict(observed_at=now,processes=processes,status=read(out/'status.json'),latest_commit=read(out/'latest.COMMITTED.json'),
    training_windows=windows,validation=validation,evaluation_error=read(out/'evaluation_error.json'),
    collector_status=read(collection/'parallel_render/status.json'),
    raw_positions=sum(read(f)['positions'] for f in collection.glob('counterfactual_*/*.json')),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
    log_tail=(r/'raw_grounding_v31.log').read_text(errors='replace')[-1500:],
    collector_log_tail=(collection/'parallel_render.log').read_text(errors='replace')[-1500:])
previous.write_text(json.dumps(result,indent=2));print(json.dumps(result))
