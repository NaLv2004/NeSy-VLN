import hashlib,json,os,signal,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_vision_v8'
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
plans=[('loss_reference','frozen_convnext','6',894263),('loss_balanced','frozen_qwen','7',894267)]
records={}
def read(p):return json.loads(p.read_text())
def command(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
def save(state):
    temporary=root/'allocation.tmp';temporary.write_text(json.dumps({'time':time.time(),'pid':os.getpid(),'state':state,'jobs':records},indent=2));temporary.replace(root/'allocation.json')
save('waiting_for_completed_evaluation')
for attempt in range(12):
    for old_name,new_name,gpu,supervisor in plans:
        if new_name in records:continue
        old=base/'symbolic_loss_v6/runs'/old_name;out=root/'runs'/new_name
        if (out/'supervisor.json').exists():raise RuntimeError('New training already present; do not launch twice')
        status=read(old/'status.json');step=status['update']
        if step%10==0 and not all((old/f'evaluations/step_{step:06d}'/suite/'summary.json').exists() for suite in ['development','fixed80']):continue
        pid=read(old/'supervisor.json').get('pid')
        assert str(old) in command(pid) and 'scripts/train.py' in command(pid) and os.getpgid(pid)==pid
        assert str(old) in command(supervisor) and 'scripts/supervise.py' in command(supervisor)
        os.killpg(pid,signal.SIGSTOP)
        try:
            status=read(old/'status.json');step=status['update'];marker=read(old/'latest/COMMITTED')
            safe=marker['epoch']==step and (step%10!=0 or all((old/f'evaluations/step_{step:06d}'/suite/'summary.json').exists() for suite in ['development','fixed80']))
            if safe:(old/'STOP').write_text('Reallocated to frozen visual-grounding comparison; all saved checkpoints and evaluation results retained.\n')
        finally:os.killpg(pid,signal.SIGCONT)
        if not safe:continue
        os.killpg(pid,signal.SIGTERM);os.kill(supervisor,signal.SIGTERM)
        deadline=time.time()+45
        while (command(pid) or command(supervisor)) and time.time()<deadline:time.sleep(1)
        if command(pid) or command(supervisor):raise RuntimeError('Owned old processes did not exit')
        marker=read(old/'latest/COMMITTED');path=old/'latest'/marker['checkpoint_file']
        assert marker['optimizer_included'] and hashlib.sha256(path.read_bytes()).hexdigest()==marker['checkpoint_sha256']
        out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        with (out/'supervisor.log').open('ab') as log:
            child=subprocess.Popen([python,'-u','scripts/supervise.py','--out',str(out),'--config',f'configs/{new_name}.json','--gpus',gpu],
                cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records[new_name]={'old_run':str(old),'old_final_checkpoint':marker,'new_run':str(out),'gpu':gpu,'supervisor_pid':child.pid,'time':time.time()}
        save('allocating');print(json.dumps(records[new_name]),flush=True)
    if len(records)==2:save('complete');break
    time.sleep(600)
else:raise RuntimeError('No safe allocation boundary found; inspect old jobs')
