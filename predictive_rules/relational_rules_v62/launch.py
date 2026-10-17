import json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_relational_rules_v62';SOURCE=ROOT/'src_relational_rules_v62'
OUT.mkdir(exist_ok=True);assert not (OUT/'launcher.json').exists(),'Inspect original handle before retrying'
assert shutil.disk_usage(ROOT).free>100_000_000
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'probe.log').open('ab',buffering=0) as log:
    process=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python',
        '-u',str(SOURCE/'probe.py')],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=process.pid,time=time.time(),device='CPU',threads=2,updates=1200,
    official_dev80_used=False,finalfixed80_used=False)
(OUT/'launcher.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
