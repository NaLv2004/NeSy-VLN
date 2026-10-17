import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'diagnostic_stable_probability_v50';out.mkdir(exist_ok=True)
launch=out/'launcher.json'
assert not launch.exists(),'Inspect existing process, do not duplicate launch'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3')
py='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
src=ROOT/'src_stable_probability_v50'
with (out/'console.log').open('ab') as log:
    child=subprocess.Popen([py,'-u',str(src/'run_diagnostic.py')],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,time=time.time(),command=Path(f'/proc/{child.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
launch.write_text(json.dumps(record,indent=2));print(json.dumps(record))
