import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'spatial_automaton_v43';out.mkdir(exist_ok=True)
script=base/'train_spatial_automaton_v43.py'
assert not (out/'launcher.json').exists() and not (out/'status.json').exists(),'Inspect existing work before any relaunch'
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    if proc.parent.name==str(os.getpid()):continue
    try:assert str(script).encode() not in proc.read_bytes(),'Existing process is live'
    except OSError:pass
free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>18000,free
assert shutil.disk_usage(base).free/2**30>1.5,'Need space for rule programs and evidence'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONPATH=str(base/'induced_boolean_rules_v41_deps'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (out/'driver.log').open('ab') as log:
    process=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script)],
        env=env,cwd=base,stdout=log,stderr=log,start_new_session=True)
record={'pid':process.pid,'time':time.time(),'script':str(script),'gpu':7,'main_training_unchanged':True,'source':'Original training demonstrations only; original80development evaluation follows once'}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
