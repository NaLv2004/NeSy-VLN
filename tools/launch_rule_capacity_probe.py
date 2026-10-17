import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'paired_goal_probe_v12';marker=out/'rule_capacity_process.json'
if marker.exists():
    previous=json.loads(marker.read_text());p=Path(f"/proc/{previous['pid']}/cmdline")
    if p.exists() and b'probe_discrete_rule_capacity.py' in p.read_bytes():print(json.dumps(previous));raise SystemExit
if (out/'rule_capacity.json').exists():raise RuntimeError('Inspect previous capacity result first')
with (out/'rule_capacity.log').open('ab') as log:
    child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',
        str(base/'probe_discrete_rule_capacity.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),start_new_session=True)
record={'pid':child.pid,'time':time.time()};marker.write_text(json.dumps(record));print(json.dumps(record))
