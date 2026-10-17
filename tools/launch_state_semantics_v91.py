import json,os,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/state_semantics_v91');b.mkdir(exist_ok=True)
assert not (b/'launcher.json').exists(),'Inspect existing process, do not duplicate'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',CUDA_VISIBLE_DEVICES='0')
with (b/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/sim/bin/python','-u',str(b/'audit.py')],env=env,stdout=log,stderr=log,start_new_session=True)
record=dict(pid=p.pid,time=time.time());(b/'launcher.json').write_text(json.dumps(record));print(json.dumps(record))
