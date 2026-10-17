import hashlib,json,os,signal,subprocess,time,ctypes,select
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_joint_v9';old=base/'symbolic_robust_v7/runs/robust_reference';out=root/'runs/joint'
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def read(p):return json.loads(p.read_text())
def command(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
def save(state,**fields):
    tmp=root/'allocation.tmp';tmp.write_text(json.dumps({'time':time.time(),'controller_pid':os.getpid(),'state':state,**fields},indent=2));tmp.replace(root/'allocation.json')
assert read(root/'integration_canonical/status.json')['state']=='complete'
test=read(root/'integration_canonical/latest/COMMITTED');assert test['optimizer_included']
row=json.loads((root/'integration_canonical/rank0.jsonl').read_text().splitlines()[-1])
assert row['episodes']>0 and row['losses']['ppo_ratio_max_error']<1e-4 and row['symbolic_audit']['replay_probability_max_error']<1e-5
assert not (out/'supervisor.json').exists()
save('waiting_for_completed_evaluation')
# Wake on an atomic saved-update event rather than repeatedly missing short
# safe boundaries with a ten-minute timer. No training-status polling loop.
libc=ctypes.CDLL(None,use_errno=True)
event_fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
assert event_fd>=0
watch=libc.inotify_add_watch(event_fd,str(old).encode(),0x80)
assert watch>=0
def wait_for_update():
    if select.select([event_fd],[],[],600)[0]:os.read(event_fd,65536)
for attempt in range(12):
    status=read(old/'status.json');step=status['update']
    if step<40 or (step%10==0 and not all((old/f'evaluations/step_{step:06d}'/suite/'summary.json').exists() for suite in ['development','fixed80'])):
        wait_for_update();continue
    pid=read(old/'supervisor.json')['pid'];supervisor=1053824
    assert str(old) in command(pid) and 'scripts/train.py' in command(pid) and os.getpgid(pid)==pid
    assert str(old) in command(supervisor) and 'scripts/supervise.py' in command(supervisor)
    os.killpg(pid,signal.SIGSTOP)
    try:
        status=read(old/'status.json');step=status['update'];marker=read(old/'latest/COMMITTED')
        safe=marker['epoch']==step and (step%10!=0 or all((old/f'evaluations/step_{step:06d}'/suite/'summary.json').exists() for suite in ['development','fixed80']))
        if safe:(old/'STOP').write_text('Reallocated to discrete joint RL after a complete saved update; committed models and validation retained.\n')
    finally:os.killpg(pid,signal.SIGCONT)
    if not safe:wait_for_update();continue
    os.killpg(pid,signal.SIGTERM);os.kill(supervisor,signal.SIGTERM)
    deadline=time.time()+45
    while (command(pid) or command(supervisor)) and time.time()<deadline:time.sleep(1)
    assert not command(pid) and not command(supervisor),'Old owned processes still live'
    marker=read(old/'latest/COMMITTED');file=old/'latest'/marker['checkpoint_file']
    assert marker['optimizer_included'] and hashlib.sha256(file.read_bytes()).hexdigest()==marker['checkpoint_sha256']
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    with (out/'supervisor.log').open('ab') as log:
        child=subprocess.Popen([python,'-u','scripts/supervise.py','--out',str(out),'--config','configs/joint.json','--gpus','0,1,2'],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save('complete',old_run=str(old),old_final_checkpoint=marker,new_run=str(out),supervisor_pid=child.pid,gpus=[0,1,2]);break
else:raise RuntimeError('No completed safe boundary; inspect existing run')
