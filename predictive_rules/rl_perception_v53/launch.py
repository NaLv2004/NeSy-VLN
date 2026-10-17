import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_rl_perception_v53';OUT.mkdir(exist_ok=True)
launch=OUT/'launcher.json'
if launch.exists():
    saved=json.loads(launch.read_text());pid=saved['pid']
    if Path(f'/proc/{pid}/cmdline').exists():raise RuntimeError('Existing process must be inspected, not duplicated')
    if (OUT/'report.json').exists():raise RuntimeError('Probe already complete')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
log=(OUT/'process.log').open('a')
process=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',
    str(ROOT/'src_rl_perception_v53/probe.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,time=time.time(),gpu=7,diagnostic_only=True)
launch.write_text(json.dumps(record,indent=2));print(json.dumps(record))
