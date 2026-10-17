"""Preserve completed evaluations before reallocating two single-card trials."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');old_root=base/'discrete_v4';root=base/'symbolic_loss_v6'
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
lock=root/'launcher.lock'
try:descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
except FileExistsError:raise RuntimeError('An allocation controller already exists; inspect its PID first')
os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
plans=[('discrete_gru','loss_reference','6',725177),('discrete_vlm','loss_balanced','7',725178)]
records={};started=time.time()

def command(pid):
    path=Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else ''

def save():
    temporary=root/'allocation.tmp';temporary.write_text(json.dumps({'time':time.time(),'pid':os.getpid(),
        'state':'complete' if len(records)==len(plans) else 'waiting_for_evaluation','jobs':records},indent=2))
    temporary.replace(root/'allocation.json')

save()
for attempt in range(12):
    for old_name,new_name,gpu,supervisor in plans:
        if new_name in records:continue
        old=old_root/'runs'/old_name;out=root/'runs'/new_name
        if (out/'supervisor.json').exists():raise RuntimeError('New training is already present')
        status=json.loads((old/'status.json').read_text());update=status['update']
        if update%10==0 and not all((old/f'evaluations/step_{update:06d}'/s/'summary.json').exists()
                for s in ['development','fixed80']):continue
        existing=json.loads((old/'supervisor.json').read_text());pid=existing.get('pid')
        if 'scripts/train.py' not in command(pid) or str(old) not in command(pid):
            raise RuntimeError('Expected training process ownership cannot be verified')
        if 'scripts/supervise.py' not in command(supervisor) or str(old) not in command(supervisor):
            raise RuntimeError('Expected supervisor ownership cannot be verified')
        marker=json.loads((old/'latest/COMMITTED').read_text())
        assert marker['optimizer_included']
        # The evaluation just checked uses committed weights. Recheck before stopping.
        if json.loads((old/'status.json').read_text())['update']!=update:continue
        assert os.getpgid(pid)==pid
        # Briefly freeze owned workers to close the race with the next commit or evaluation.
        os.killpg(pid,signal.SIGSTOP)
        try:
            frozen=json.loads((old/'status.json').read_text())
            committed=json.loads((old/'latest/COMMITTED').read_text())
            step=frozen['update']
            safe=committed['epoch']==step and (step%10!=0 or all(
                (old/f'evaluations/step_{step:06d}'/s/'summary.json').exists() for s in ['development','fixed80']))
            if safe:(old/'STOP').write_text('Reallocated to matched symbolic transition-loss SFT trials; saved evidence retained.\n')
        finally:os.killpg(pid,signal.SIGCONT)
        if not safe:continue
        os.killpg(pid,signal.SIGTERM)
        os.kill(supervisor,signal.SIGTERM)
        deadline=time.time()+45
        while (command(pid) or command(supervisor)) and time.time()<deadline:time.sleep(1)
        if command(pid) or command(supervisor):raise RuntimeError('Old processes did not exit')
        marker=json.loads((old/'latest/COMMITTED').read_text())
        checkpoint=old/'latest'/marker['checkpoint_file']
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest==marker['checkpoint_sha256']
        out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
        with (out/'supervisor.log').open('ab') as log:
            process=subprocess.Popen([python,'-u','scripts/supervise.py','--out',str(out),
                '--config',f'configs/{new_name}.json','--gpus',gpu],cwd=root,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records[new_name]={'old_run':str(old),'old_final':marker,'new_run':str(out),'gpu':gpu,
            'supervisor_pid':process.pid,'time':time.time()};save()
        print(json.dumps(records[new_name]),flush=True)
    if len(records)==len(plans):break
    time.sleep(600)
else:raise RuntimeError('No safe allocation boundary found; old training was not forcibly interrupted')
