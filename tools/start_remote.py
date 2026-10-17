import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010');run=root/'runs/full_v1';run.mkdir(parents=True,exist_ok=True)
status=run/'supervisor.json'
if status.exists():
    previous=json.loads(status.read_text());pid=previous.get('pid')
    if pid and Path(f'/proc/{pid}/cmdline').exists():
        command=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        if 'scripts/train.py' in command:raise RuntimeError('This run is already active')
env=dict(os.environ,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
(run/'STOP').unlink(missing_ok=True)
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
with (run/'supervisor.log').open('ab') as log:
    child=subprocess.Popen([python,'-u','scripts/supervise.py','--out',str(run),'--gpus','0,1,2,3,4,5'],
        cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':child.pid,'run':str(run),'started':time.time()}))
