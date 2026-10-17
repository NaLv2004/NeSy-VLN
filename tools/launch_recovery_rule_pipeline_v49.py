import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'recovery_rule_learning_v49';out.mkdir(exist_ok=True);assert not (out/'launcher.json').exists()
src=out/'src';src.mkdir(exist_ok=True);shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
for origin,name in [('merge_onpolicy_recovery_v48.py','merge.py'),('encode_onpolicy_recovery_v48.py','encode.py'),('probe_recovery_rules_v49.py','probe.py'),('run_recovery_rule_pipeline_v49.py','pipeline.py')]:shutil.copy2(b/origin,src/name)
with (out/'pipeline.log').open('ab') as log:p=subprocess.Popen(['/usr/bin/python3','-u',str(src/'pipeline.py')],cwd=src,stdout=log,stderr=log,start_new_session=True)
row={'pid':p.pid,'time':time.time(),'script':str(src/'pipeline.py'),'scripts_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in src.glob('*.py')},'dependency':'Wait on actual v48 collector pidfds, validate successes/splits/images, freeze visual codes, run matched original80development trials.','automatic_restart':False};(out/'launcher.json').write_text(json.dumps(row,indent=2));print(json.dumps(row,indent=2))
