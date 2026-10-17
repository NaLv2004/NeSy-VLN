import json,shutil,subprocess,time
from pathlib import Path
OUT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/counterfactual_closure_v63')
def read(p):return json.loads(p.read_text()) if p.exists() else {}
prior=read(OUT/'last_observation.json')
launch=read(OUT/'launcher.json')
if prior and prior.get('launch',{}).get('pid')==launch.get('pid') and time.time()<prior['time']+630:
    print(json.dumps(dict(state='not_due',last_check=prior['time'],next_check=prior['time']+630)))
    raise SystemExit(0)
status=read(OUT/'status.json')
ranks=[read(OUT/f'status_rank{r}.json') for r in (0,1)]
run=OUT.parent/'run_counterfactual_closure_v63'
handoff=read(run/'launcher.json');supervisor=read(run/'supervisor.json')
pids={launch['pid'],*status.get('workers',[]),handoff.get('pid'),supervisor.get('worker_pid')}-{None};processes=[]
for pid in sorted(pids):
    p=Path(f'/proc/{pid}/cmdline');processes.append(dict(pid=pid,command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''))
samples=[]
for p in sorted((OUT/'episodes').glob('*.json'))[:2]:
    r=read(p);samples.append({k:r[k] for k in ('episode','positions','executed_actions','two_step_probe_execution_errors','offline_future_labels_never_assigned_to_policy')})
logs={}
for name in ('collector_rank0.log','collector_rank1.log','coordinator.log'):
    p=OUT/name
    if p.exists():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1600));logs[name]=f.read().decode(errors='replace')
result=dict(time=time.time(),launch=launch,status=status,ranks=ranks,processes=processes,samples=samples,logs=logs,
    handoff=handoff,training_supervisor=supervisor,
    training_ranks=[read(run/f'status_rank{r}.json') for r in (0,1)],
    free_bytes=shutil.disk_usage(OUT).free,
    gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
(OUT/'last_observation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
