import os,json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/v2');out=root/'runs/full_v2';out.mkdir(parents=True,exist_ok=True)
status=out/'supervisor.json'
if status.exists():
    s=json.loads(status.read_text());pid=s.get('pid')
    if pid and Path(f'/proc/{pid}/cmdline').exists():
        command=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        if 'scripts/train.py' in command:raise RuntimeError('The v2 run is already active')
integration=json.loads((root/'runs/integration/latest/COMMITTED').read_text())
assert integration['epoch']==1
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
         HF_HUB_OFFLINE='1',TRITON_CACHE_DIR=str(root/'cache/triton'))
(out/'STOP').unlink(missing_ok=True)
with (out/'supervisor.log').open('ab') as log:
    child=subprocess.Popen([python,'-u','scripts/supervise.py','--out',str(out),
        '--config','configs/navigation_v2.json','--gpus','0,1,2,3,4,5'],
        cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':child.pid,'out':str(out),'started':time.time()}))
