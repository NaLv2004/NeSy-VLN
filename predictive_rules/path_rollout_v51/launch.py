import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'diagnostic_path_rollout_v51';src=ROOT/'src_path_rollout_v51';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists()
prior=json.loads((ROOT/'diagnostic_stable_probability_v50/launcher.json').read_text())
assert not Path(f'/proc/{prior["pid"]}/cmdline').exists(),'Wait for numerical diagnostic to finish'
assert (ROOT/'diagnostic_stable_probability_v50/report.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3')
py='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
with (out/'console.log').open('ab') as log:
    child=subprocess.Popen([py,'-u',str(src/'run_diagnostic.py')],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,time=time.time());(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
