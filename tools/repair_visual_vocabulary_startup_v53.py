import json,os,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'visual_vocabulary_probe_v53';old=json.loads((out/'launcher.json').read_text())
assert not Path(f'/proc/{old["pid"]}').exists(),'Do not duplicate a living process'
assert not (out/'result.json').exists() and not (out/'manifest.json').exists()
assert "ModuleNotFoundError: No module named 'scipy'" in (out/'driver.log').read_text()
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONPATH=str(b/'induced_boolean_rules_v41_deps'))
subprocess.run([policy,'-c','import scipy,sklearn;print(scipy.__version__,sklearn.__version__)'],env=env,check=True)
(out/'failed_startup_launcher.json').write_text(json.dumps(old,indent=2))
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen([policy,'-u',str(out/'src/probe.py')],env=env,cwd=out/'src',stdout=log,stderr=log,start_new_session=True)
new=dict(old,pid=p.pid,time=time.time(),repair='Restore existing scipy/sklearn dependency path; original process missing, no fitting or evaluation started.',previous_pid=old['pid'])
(out/'launcher.json').write_text(json.dumps(new,indent=2));print(json.dumps(new,indent=2))
