import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'diagnostic_stable_probability_v50';src=ROOT/'src_stable_probability_v50'
prior=json.loads((out/'launcher.json').read_text())
assert not Path(f'/proc/{prior["pid"]}/cmdline').exists(),'Prior process is still alive'
assert not (out/'report.json').exists()
assert not (out/'physical/rank0.episodes.json').exists(),'Do not repeat completed episodes'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3')
py='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
with (out/'console.log').open('ab') as log:
    child=subprocess.Popen([py,'-u',str(src/'run_diagnostic.py')],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,time=time.time(),prior=prior,reason='Removed obsolete discount variable in diagnostic exporter; no physical episode had finished; reuse completed94root audit')
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
