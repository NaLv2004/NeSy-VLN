import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'data/paired_goals_v12'
out.mkdir(parents=True,exist_ok=True);marker=out/'process.json'
if marker.exists():
    previous=json.loads(marker.read_text());p=Path(f"/proc/{previous['pid']}/cmdline")
    if p.exists() and b'collect_paired_goal_demonstrations.py' in p.read_bytes():
        print(json.dumps({'already_live':previous}));raise SystemExit
if (out/'COMPLETE').exists():print((out/'COMPLETE').read_text());raise SystemExit
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/sim/bin/python','-u',
    str(base/'collect_paired_goal_demonstrations.py'),'--reference-root','/home/user/xjy/strips_wm_navigation_20261009',
    '--out',str(out),'--target-pairs','64','--gpu','7']
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')
with (out/'collector.log').open('ab') as log:
    job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
record={'pid':job.pid,'time':time.time(),'command':cmd};marker.write_text(json.dumps(record,indent=2))
print(json.dumps(record))
