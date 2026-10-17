import json,time,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_sft_v5');saved=root/'current_status.json'
if saved.exists():
    previous=json.loads(saved.read_text())
    if previous.get('healthy') and time.time()-previous['time']<600:
        print(json.dumps({'skipped':'stable training minimum 600 seconds','last_time':previous['time']}));raise SystemExit(0)
jobs={}
for name in ['sft_plain','sft_spatial']:
    out=root/'runs'/name;record={}
    for key in ['supervisor','status','initialization','architecture']:
        p=out/f'{key}.json'
        if p.exists():record[key]=json.loads(p.read_text())
    p=out/'latest/COMMITTED'
    if p.exists():record['checkpoint']=json.loads(p.read_text())
    pid=record.get('supervisor',{}).get('pid');p=Path(f'/proc/{pid}/cmdline')
    record['live_command']=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
    record['ranks']=[]
    for rank in range(3):
        p=out/f'rank{rank}.jsonl'
        if p.exists():record['ranks'].append(json.loads(p.read_text().splitlines()[-1]))
    logs=sorted(out.glob('launch*.log'),key=lambda p:p.stat().st_mtime)
    if logs:record['log_tail']=logs[-1].read_text(errors='replace')[-4000:]
    record['evaluations']={}
    for step in sorted((out/'evaluations').glob('step_*'))[-2:]:
        record['evaluations'][step.name]={suite:json.loads((step/suite/'summary.json').read_text())
            for suite in ['development','fixed80'] if (step/suite/'summary.json').exists()}
    jobs[name]=record
healthy=all(r.get('live_command') and r.get('status',{}).get('update',0)>0 and r.get('supervisor',{}).get('state')=='running' for r in jobs.values())
result={'time':time.time(),'healthy':healthy,'jobs':jobs,
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
    'disk':subprocess.check_output(['df','-h',str(root)],text=True)}
saved.write_text(json.dumps(result,indent=2))
compact={}
for name,record in jobs.items():
    compact[name]={k:record.get(k) for k in ['status','supervisor','live_command','checkpoint','evaluations']}
    compact[name]['evaluations']={step:{suite:{k:value.get(k) for k in ['episodes','successes','success_rate','collision_action_rate','mean_steps']}
        for suite,value in suites.items()} for step,suites in record['evaluations'].items()}
    compact[name]['ranks']=[{**{k:rank.get(k) for k in ['update','rank','seconds','training_frames','gpu_peak_gib']},
        'losses':{k:rank.get('losses',{}).get(k) for k in ['bc','expert_action_accuracy','transition_bit_accuracy',
            'copy_bit_accuracy','changed_symbol_bit_accuracy','inverse_accuracy']}} for rank in record['ranks']]
    if not record.get('live_command') or 'Traceback' in record.get('log_tail',''):
        compact[name]['log_tail']=record.get('log_tail')
print(json.dumps(dict(result,jobs=compact),indent=2))
