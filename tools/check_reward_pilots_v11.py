import json,time,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_reward_v11');saved=root/'current_status.json'
def read(path):return json.loads(path.read_text()) if path.exists() else None
previous=read(saved)
if previous and previous.get('stable') and time.time()-previous['time']<600:
    print(json.dumps({'skipped':'stable training minimum600 seconds','last_time':previous['time']}));raise SystemExit
result={'time':time.time(),'free_gib':shutil.disk_usage(root).free/2**30,'jobs':{}}
for record in read(root/'pilot_processes.json'):
    out=Path(record['out']);cmd=Path(f"/proc/{record['pid']}/cmdline")
    item={'process':record,'live':cmd.read_bytes().replace(b'\0',b' ').decode() if cmd.exists() else None,
        'status':read(out/'status.json'),'checkpoint':read(out/'latest/COMMITTED')}
    log=out/'rank0.jsonl'
    if log.exists():
        row=json.loads(log.read_text().splitlines()[-1]);item['last_update']={k:row[k] for k in ['update','seconds','episodes','successes','losses','symbolic_audit','gpu_peak_gib','gradient_by_module']}
    item['evaluations']={step.name:{s:read(step/s/'summary.json') for s in ['development','fixed80']} for step in (out/'evaluations').glob('step_*')}
    item['tail']=(out/'pilot.log').read_text(errors='replace')[-1000:]
    result['jobs'][record['name']]=item
result['stable']=all(j['live'] and (j['status'] or {}).get('update',0)>0 for j in result['jobs'].values())
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
saved.write_text(json.dumps(result,indent=2))
print(json.dumps({'time':result['time'],'stable':result['stable'],'free_gib':result['free_gib'],'gpu':result['gpu'],
    'jobs':{name:{'live':j['live'],'status':j['status'],'checkpoint':j['checkpoint'],
        'metrics':{k:j.get('last_update',{}).get(k) for k in ['seconds','episodes','successes','gpu_peak_gib']},
        'ppo_ratio_max_error':j.get('last_update',{}).get('losses',{}).get('ppo_ratio_max_error'),
        'evaluations':j['evaluations'],'tail':j['tail'] if not j['live'] else None} for name,j in result['jobs'].items()}},indent=2))
