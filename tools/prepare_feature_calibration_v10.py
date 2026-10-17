import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/calibrated_vision_v10')
assert not (root/'calibration_process.json').exists()
assert '59 passed' in (root/'verification.log').read_text()
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
with (root/'calibration.log').open('ab') as log:
    p=subprocess.Popen([python,'-u','scripts/fit_frozen_feature_calibration.py'],cwd=root,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,HF_HUB_OFFLINE='1'))
(root/'calibration_process.json').write_text(json.dumps({'pid':p.pid,'time':time.time()},indent=2));print(json.dumps({'pid':p.pid}))
