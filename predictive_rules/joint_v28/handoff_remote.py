"""Replace only our v27 trainer, allowing its active evaluation to finish.

Waiting uses kernel process-exit notification rather than frequent polling.
All unrelated processes and all committed checkpoints are preserved.
"""
import hashlib,json,os,select,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
STATE=ROOT/'joint_observed_v28_handoff.json'
OLD=1257123


def record(**values):
    values.update(time=time.time(),pid=os.getpid());temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(values,indent=2));os.replace(temporary,STATE)


def command(pid):
    path=Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else ''


def wait_exit(fd,timeout):
    return bool(select.select([fd],[],[],timeout)[0])


def process_descriptor(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    # Python 3.8 lacks the wrapper, although the Linux kernel supports it.
    import ctypes,platform,re
    assert platform.system()=='Linux' and platform.machine()=='x86_64'
    header=Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    number=int(re.search(r'#define\s+__NR_pidfd_open\s+(\d+)',header).group(1))
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(number,pid,0)
    if fd<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return fd


def worker():
    out=ROOT/'run_joint_observed_v28';source=ROOT/'src_joint_observed_v28'
    assert not out.exists()
    checkpoints=sorted((ROOT/'run_adaptive_observed_v27').glob('checkpoint_*.COMMITTED.json'))
    assert checkpoints
    commit=checkpoints[-1];checkpoint=commit.with_name(commit.name.replace('.COMMITTED.json','.pt'))
    expected=json.loads(commit.read_text())['sha256']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    original=command(OLD);parentfd=None
    if original:
        assert str(ROOT/'src_adaptive_observed_v27/train.py') in original and str(ROOT/'run_adaptive_observed_v27') in original
        parentfd=process_descriptor(OLD);os.kill(OLD,signal.SIGSTOP)
        try:
            children=[]
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():continue
                try:
                    status=(path/'status').read_text()
                    ppid=int(next(line for line in status.splitlines() if line.startswith('PPid:')).split()[1])
                    if ppid!=OLD:continue
                    pid=int(path.name);cmd=command(pid)
                    if not cmd:continue
                    assert str(ROOT/'src_adaptive_observed_v27/evaluate.py') in cmd,cmd
                    children.append((pid,process_descriptor(pid)))
                except (FileNotFoundError,ProcessLookupError):continue
            record(state='waiting_for_existing_evaluation',old_trainer=OLD,evaluators=[p for p,_ in children],
                   preserved_checkpoint=str(checkpoint),preserved_sha256=expected)
            for pid,fd in children:
                # No repeated remote status polling. The descriptor signals
                # only when this exact evaluation process exits.
                if not wait_exit(fd,3600):raise TimeoutError(f'Evaluation {pid} did not finish')
                os.close(fd)
            evaluation=ROOT/'run_adaptive_observed_v27'/('development_step_'+checkpoint.stem.split('_')[-1])/'summary.json'
            if evaluation.exists():
                result=json.loads(evaluation.read_text())
                if result['successes']>=48 and result['executed_steps_per_program']>=5:
                    os.kill(OLD,signal.SIGCONT)
                    record(state='candidate_goal_met_in_previous_run',checkpoint=str(checkpoint),summary=result)
                    return
            if command(OLD):
                assert command(OLD)==original
                os.kill(OLD,signal.SIGTERM);os.kill(OLD,signal.SIGCONT)
                if not wait_exit(parentfd,30):raise TimeoutError('Owned trainer did not terminate')
        except Exception:
            if command(OLD)==original:os.kill(OLD,signal.SIGCONT)
            raise
        finally:os.close(parentfd)
        status_path=ROOT/'run_adaptive_observed_v27/status.json'
        previous=json.loads(status_path.read_text())
        previous.update(state='stopped_for_joint_representation_training',stopped_time=time.time(),
                        successor_run=str(out),checkpoints_preserved=True,goal_complete=False)
        temporary=status_path.with_suffix('.tmp');temporary.write_text(json.dumps(previous,indent=2));temporary.replace(status_path)
    used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>256:raise RuntimeError(f'GPU7 has {used} MiB occupied; preserving the other process')
    args=[PYTHON,'-u',str(source/'train.py'),'--source',str(checkpoint),
          '--out',str(out),'--batch-size','64','--updates','2500','--eval-every','500']
    with (ROOT/'joint_observed_v28.log').open('xb',buffering=0) as log:
        job=subprocess.Popen(args,cwd=source,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(pid=job.pid,command=args,gpu=7,created=time.time(),out=str(out),source=str(source),goal_complete=False)
    (ROOT/'joint_observed_v28_launcher.json').write_text(json.dumps(launch,indent=2))
    record(state='launched',training=launch,preserved_checkpoint=str(checkpoint),preserved_sha256=expected)


if __name__=='__main__':
    if '--worker' in sys.argv:
        try:worker()
        except Exception as error:
            import traceback
            record(state='failed',error=repr(error),traceback=traceback.format_exc());raise
    else:
        if STATE.exists():
            previous=json.loads(STATE.read_text())
            assert '--retry' in sys.argv and previous['state']=='failed' and not command(previous['pid'])
            STATE.rename(STATE.with_suffix(f'.failed_{int(time.time())}.json'))
        with (ROOT/'joint_observed_v28_handoff.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-u',__file__,'--worker'],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(handoff_pid=process.pid,state_file=str(STATE))))
