import json,time,subprocess,shutil
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');saved=base/'research_monitor_20261011.json'
previous=json.loads(saved.read_text()) if saved.exists() else json.loads((base/'storage_recovery_verification.json').read_text())
startup=base/'symbolic_task_v15/startup_inspection.json'
last=max(previous['time'],json.loads(startup.read_text())['time'] if startup.exists() else 0)
branch_start=base/'symbolic_branch_v17/startup_inspection.json'
if branch_start.exists():last=max(last,json.loads(branch_start.read_text())['time'])
conjunctive_start=base/'symbolic_conjunctive_v23/startup_inspection.json'
if conjunctive_start.exists():last=max(last,json.loads(conjunctive_start.read_text())['time'])
teacher_start=base/'symbolic_teacher_v25/startup_inspection.json'
if teacher_start.exists():last=max(last,json.loads(teacher_start.read_text())['time'])
recovery_start=base/'symbolic_recovery_v28/startup_inspection.json'
if recovery_start.exists():last=max(last,json.loads(recovery_start.read_text())['time'])
neural_start=base/'symbolic_execution_v38/startup_inspection.json'
if neural_start.exists():last=max(last,json.loads(neural_start.read_text())['time'])
repair=base/'storage_repair_v34.json'
if repair.exists():last=max(last,json.loads(repair.read_text())['time'])
repair46=base/'storage_repair_verification_v46.json'
if repair46.exists():last=max(last,json.loads(repair46.read_text())['time'])
if time.time()-last<600:
    print(json.dumps({'skipped':'minimum600seconds','next_check':last+600}));raise SystemExit
def read(path):return json.loads(path.read_text()) if path.exists() else None
def live(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for relative in ['symbolic_robust_v7/runs/robust_multieffect','symbolic_joint_v9/runs/joint',
    'symbolic_vision_v8/runs/frozen_convnext','symbolic_vision_v8/runs/frozen_qwen',
    'symbolic_reward_v11/runs/original','symbolic_reward_v11/runs/bounded_repeat_stop40',
    'symbolic_task_v15/runs/paired_tasks','symbolic_branch_v17/runs/branch_current',
    'symbolic_conjunctive_v23/runs/branch_control','symbolic_conjunctive_v23/runs/conjunctive',
    'symbolic_teacher_v25/runs/ema_control','symbolic_teacher_v25/runs/synchronous',
    'symbolic_recovery_v28/runs/base_control','symbolic_recovery_v28/runs/recovery_mixture',
    'symbolic_execution_v38/runs/detached_outcome_control','symbolic_execution_v38/runs/grounded_outcome_feedback']:
    if not (base/relative).exists():continue
    out=base/relative;supervisor=read(out/'supervisor.json');pid=(supervisor or {}).get('pid')
    if 'symbolic_reward_v11' in relative:
        pid=next(r['pid'] for r in read(base/'symbolic_reward_v11/pilot_processes.json') if r['name']==out.name)
    record={'status':read(out/'status.json'),'commit':read(out/'latest/COMMITTED'),'retired':(out/'STOP').exists(),
        'pid':pid,'live':live(pid),'supervisor':supervisor,'storage_wait':read(out/'latest/WAITING_FOR_STORAGE.json')}
    record['storage_waits']={str(p.relative_to(out)):read(p) for p in out.rglob('WAITING_FOR_STORAGE.json')}
    lines=(out/'rank0.jsonl').read_text().splitlines() if (out/'rank0.jsonl').exists() else []
    record['last_update']=json.loads(lines[-1]) if lines else None
    record['evaluations']={step.name:{s:read(step/s/'summary.json') for s in ['development','fixed80']}
        for step in sorted((out/'evaluations').glob('step_*'))}
    jobs[relative]=record
result={'time':time.time(),'jobs':jobs,'free_gib':shutil.disk_usage(base).free/2**30,
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
saved.write_text(json.dumps(result,indent=2))
def compact_evaluation(evaluations):
    if not evaluations:return None
    name=list(evaluations)[-1];value=evaluations[name]
    return {'step':name,**{suite:({k:summary[k] for k in ['episodes','successes','collision_action_rate','mean_steps']} if summary else None)
                          for suite,summary in value.items()}}
print(json.dumps({'time':result['time'],'free_gib':result['free_gib'],'gpu':result['gpu'],'jobs':{
    name:{'status':j['status'],'commit':j['commit'],'live':j['live'],'attempt':(j['supervisor'] or {}).get('attempt'),
        'storage_wait':j['storage_wait'],'storage_waits':j['storage_waits'],'ppo_ratio_error':(j['last_update'] or {}).get('losses',{}).get('ppo_ratio_max_error'),
        'latest_evaluation':compact_evaluation(j['evaluations'])}
    for name,j in jobs.items() if any(tag in name for tag in ['_v23/','_v25/','_v28/','_v38/'])}},indent=2))
