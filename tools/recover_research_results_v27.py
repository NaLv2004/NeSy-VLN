import json,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010')
plan=json.loads((base/'retired_evaluation_pruning_plan.json').read_text())
print(json.dumps({'plan_candidates':len(plan['items']),'reclaimable_gib':sum(r['reclaimable_bytes'] for r in plan['items'])/2**30,
                 'candidate_folders':[r['folder'] for r in plan['items']]}))
print((base/'symbolic_paired_v12/recovery_regression_results.log').read_text()[-1500:])
saved=base/'symbolic_conjunctive_v23/completion_inspection.json'
if saved.exists():
    s=json.loads(saved.read_text());print(json.dumps({'completion_time':s['time'],'pending':s['pending'],
        'jobs':{k:{kk:vv for kk,vv in v.items() if kk!='evaluations'} for k,v in s['jobs'].items()}}))
for family in ['symbolic_conjunctive_v23','symbolic_teacher_v25']:
    for out in sorted((base/family/'runs').iterdir()):
        if not out.is_dir():continue
        sup=json.loads((out/'supervisor.json').read_text());pid=sup['pid'];proc=Path(f'/proc/{pid}/cmdline')
        print(json.dumps({'run':str(out),'status':json.loads((out/'status.json').read_text()),'supervisor':sup,
            'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
            'storage_waits':{str(p.relative_to(out)):json.loads(p.read_text()) for p in out.rglob('WAITING_FOR_STORAGE.json')}}))
print(json.dumps({'time':time.time(),'free_gib':shutil.disk_usage(base).free/2**30,
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}))
