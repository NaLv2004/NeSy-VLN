import json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_reward_v11')
assert not (root/'integration_process.json').exists()
assert '67 passed' in (root/'verification.log').read_text()
assert shutil.disk_usage(root).free>2*2**30
with (root/'integration.log').open('ab') as log:
    process=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(root/'integration.py')],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1'))
record={'pid':process.pid,'time':time.time(),'gpu':7,'memory_fraction':.30}
(root/'integration_process.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
