"""Run the CPU-only diagnostic when the already-running collector commits data."""
import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');data=base/'data/paired_goals_v12';out=base/'paired_goal_probe_v12'
out.mkdir(exist_ok=True);status=out/'controller_status.json'
while not (data/'COMPLETE').exists():
    handle=json.loads((data/'process.json').read_text());p=Path(f"/proc/{handle['pid']}/cmdline")
    live=p.exists() and b'collect_paired_goal_demonstrations.py' in p.read_bytes()
    status.write_text(json.dumps({'state':'waiting_for_committed_data' if live else 'collector_missing',
        'time':time.time(),'pid':os.getpid(),'collector_pid':handle['pid'],'collector_live_verified':live,'poll_seconds':600}))
    if not live:raise RuntimeError('Collector ended without a complete dataset; do not silently restart')
    time.sleep(600)
if (out/'result.json').exists() and json.loads((out/'result.json').read_text()).get('state')=='complete':raise SystemExit
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(base/'probe_paired_goal_learning.py'),
    '--root',str(base/'symbolic_reward_v11'),'--data',str(data),
    '--checkpoint',str(base/'symbolic_reward_v11/initialization_source'),'--out',str(out),'--steps','160']
status.write_text(json.dumps({'state':'running_probe','time':time.time(),'pid':os.getpid(),'command':cmd}))
with (out/'probe.log').open('ab') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    (out/'probe_process.json').write_text(json.dumps({'pid':child.pid,'command':cmd,'time':time.time()}));code=child.wait()
status.write_text(json.dumps({'state':'complete' if code==0 else 'failed','returncode':code,'time':time.time(),'pid':os.getpid()}))
