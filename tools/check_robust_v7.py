import json,subprocess,time,shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_robust_v7');saved=root/'current_status.json'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def live(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
previous=read(saved)
if previous and previous.get('stable') and time.time()-previous['time']<600:
    print(json.dumps({'skipped':'stable training minimum600 seconds','last_time':previous['time']}));raise SystemExit
controller=read(root/'controller.json');result={'time':time.time(),'controller':controller,
    'controller_live':live(controller['pid']),'allocation':read(root/'allocation.json'),'jobs':{}}
for name in ['robust_reference','robust_multieffect']:
    out=root/'runs'/name;supervisor=read(out/'supervisor.json')
    record={k:read(out/f'{k}.json') for k in ['status','initialization','data_provenance']}
    record.update(supervisor=supervisor,live_command=live((supervisor or {}).get('pid')),checkpoint=read(out/'latest/COMMITTED'))
    record['intentionally_retired']=(out/'STOP').exists() and not record['live_command']
    record['ranks']=[]
    for p in sorted(out.glob('rank*.jsonl')):
        lines=p.read_text().splitlines()
        if lines:
            last=json.loads(lines[-1]);record['ranks'].append({k:last[k] for k in ['update','rank','seconds','training_frames','gpu_peak_gib','losses','gradient_by_module']})
    record['evaluations']={}
    for folder in sorted((out/'evaluations').glob('step_*'))[-2:]:
        record['evaluations'][folder.name]={suite:read(folder/suite/'summary.json') for suite in ['development','fixed80']}
    if not record['intentionally_retired'] and (not record['status'] or not record['live_command']):
        logs=sorted(out.glob('launch*.log'),key=lambda p:p.stat().st_mtime)
        if logs:record['startup_error']=logs[-1].read_text(errors='replace')[-2500:]
    result['jobs'][name]=record
result['stable']=all(x['intentionally_retired'] or (x['live_command'] and (x['status'] or {}).get('update',0)>0) for x in result['jobs'].values())
result['free_bytes']=shutil.disk_usage(root).free
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
result['controller_tail']=(root/'controller.log').read_text(errors='replace')[-1500:]
saved.write_text(json.dumps(result,indent=2))
compact={k:result[k] for k in ['time','stable','free_bytes','gpu','controller_live'] if k in result}
compact['jobs']={name:{'status':j['status'],'live':j['live_command'],'checkpoint':j['checkpoint'],
    'losses':[{k:r['losses'].get(k) for k in ['expert_action_accuracy','inverse_accuracy','changed_symbol_bit_accuracy','view_bit_disagreement_metric']} for r in j['ranks']],
    'evaluations':{step:{suite:({k:v.get(k) for k in ['successes','episodes','mean_steps','collision_action_rate']} if v else None)
    for suite,v in suites.items()} for step,suites in j['evaluations'].items()}} for name,j in result['jobs'].items()}
print(json.dumps(compact,indent=2))
