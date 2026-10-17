import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_teacher_v25'
assert not (root/'processes.json').exists(),'Do not duplicate existing processes'
old=base/'symbolic_branch_v17/runs/branch_current'
status=json.loads((old/'status.json').read_text())
assert status['state']=='complete'
for path in Path('/proc').glob('[0-9]*/cmdline'):
    if path.parent.name==str(os.getpid()):continue
    try:command_line=path.read_bytes()
    except OSError:continue
    assert str(old).encode() not in command_line,'Previous trainer must actually be terminal'
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(a):int(b) for a,b in (line.split(',') for line in memory.splitlines())}
assert all(free[g]>21000 for g in [6,7]),'Do not displace other work'
assert shutil.disk_usage(base).free>4.5*2**30
records=[]
for name,gpu in [('ema_control','6'),('synchronous','7')]:
    out=root/'runs'/name;out.mkdir(parents=True)
    command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','scripts/supervise.py',
        '--config',str(root/'configs'/f'{name}.json'),'--out',str(out),'--gpus',gpu]
    with (out/'supervisor.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    records.append({'name':name,'supervisor_pid':child.pid,'time':time.time(),'command':command,'gpus':gpu})
    (root/'processes.json').write_text(json.dumps(records,indent=2))
experiment=json.loads((root/'experiment.json').read_text());experiment['state']='launched'
(root/'experiment.json').write_text(json.dumps(experiment,indent=2));print(json.dumps(records,indent=2))
