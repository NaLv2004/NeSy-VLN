import json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'neural_conjunction_learning_v37'
read=lambda p:json.loads(p.read_text()) if p.exists() else None
saved=out/'inspection.json'
if saved.exists() and time.time()-read(saved)['time']<600:
    print(json.dumps({'skipped':'minimum600seconds','next_check':read(saved)['time']+600}));raise SystemExit
launcher=read(out/'launcher.json');proc=Path(f'/proc/{launcher["pid"]}/cmdline')
r=read(out/'result.json')
result={'time':time.time(),'launcher':launcher,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
    'result':r,'tail':(out/'driver.log').read_text(errors='replace')[-3000:]}
saved.write_text(json.dumps(result,indent=2))
print(json.dumps({**result,'result':({k:v for k,v in r.items() if k not in ['provenance','arms']}|{
    'arms':{name:{'initial':arm.get('initial'),'latest':arm.get('learning',[])[-1:] ,
        'initial_gate_gradient_norm':arm.get('initial_gate_gradient_norm'),
        'same_weights_gates_disabled':arm.get('same_weights_gates_disabled')} for name,arm in r.get('arms',{}).items()}}) if r else None},indent=2))
