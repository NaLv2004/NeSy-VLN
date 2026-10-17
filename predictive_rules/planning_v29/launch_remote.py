"""Wait for the exact owned v28 process to finish, then launch one v29 run."""
import ctypes,fcntl,hashlib,json,os,platform,re,select,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
STATE=ROOT/'planning_observed_v29_handoff.json'
OLD=1307827

def record(**value):
    value.update(time=time.time(),pid=os.getpid())
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(STATE)

def command(pid):
    path=Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else ''

def descriptor(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    assert platform.system()=='Linux' and platform.machine()=='x86_64'
    header=Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    number=int(re.search(r'#define\s+__NR_pidfd_open\s+(\d+)',header).group(1))
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(number,pid,0)
    if fd<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return fd

def worker():
    lock=(ROOT/'planning_observed_v29_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'run_planning_observed_v29';source=ROOT/'src_planning_observed_v29'
    assert not out.exists(),'Do not launch another trainer into an existing run'
    original=command(OLD)
    if original:
        assert str(ROOT/'src_joint_observed_v28/train.py') in original and str(ROOT/'run_joint_observed_v28') in original
        fd=descriptor(OLD);record(state='waiting_for_owned_v28_completion',old_pid=OLD,old_command=original)
        try:
            if not select.select([fd],[],[],3600)[0]:raise TimeoutError('Still running; do not terminate or duplicate it')
        finally:os.close(fd)
    previous=ROOT/'run_joint_observed_v28';summaries=[]
    for path in sorted(previous.glob('development_step_*/summary.json')):
        result=json.loads(path.read_text());summaries.append(dict(path=str(path),**result))
    assert summaries,'Preserve and inspect completed v28 evidence before replacement'
    if any(r['successes']>=48 and r['executed_steps_per_program']>=5 for r in summaries):
        record(state='previous_candidate_requires_completion_audit',summaries=summaries);return
    checkpoint=ROOT/'run_adaptive_observed_v27/checkpoint_001000.pt'
    commit=json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']
    used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>256:raise RuntimeError(f'GPU7 still has {used} MiB occupied; do not disturb another process')
    args=[PYTHON,'-u',str(source/'train.py'),'--source',str(checkpoint),'--out',str(out),
          '--batch-size','64','--updates','3000','--eval-every','500']
    with (ROOT/'planning_observed_v29.log').open('xb',buffering=0) as log:
        job=subprocess.Popen(args,cwd=source,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2'),
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launched=dict(pid=job.pid,command=args,gpu=7,created=time.time(),out=str(out),source=str(source))
    (ROOT/'planning_observed_v29_launcher.json').write_text(json.dumps(launched,indent=2))
    record(state='launched',training=launched,source_checkpoint=str(checkpoint),source_sha256=commit['sha256'],
           prior_results=summaries,all_prior_checkpoints_preserved=True,goal_complete=False)

if __name__=='__main__':
    if '--worker' in sys.argv:
        try:worker()
        except BaseException as error:
            import traceback
            record(state='failed',error=repr(error),traceback=traceback.format_exc());raise
    else:
        assert not STATE.exists(),'Inspect the existing handoff; never blindly restart'
        with (ROOT/'planning_observed_v29_handoff.log').open('xb',buffering=0) as log:
            p=subprocess.Popen([sys.executable,'-u',__file__,'--worker'],stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(handoff_pid=p.pid,state_file=str(STATE))))
