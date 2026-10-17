"""Launch one bounded, CPU-only planning diagnostic without touching trainers."""
import json
from pathlib import Path
from predictive_remote import connect

ROOT = '/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
PYTHON = '/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'


def main():
    client = connect()
    local = Path(__file__).resolve().parents[1] / 'predictive_rules'
    try:
        source = ROOT + '/src_planning_pilot'
        _, out, err = client.exec_command('mkdir -p ' + source + '/predictive_nesy')
        if out.channel.recv_exit_status():
            raise RuntimeError(err.read().decode())
        with client.open_sftp() as ftp:
            for relative in ['pilot_planning.py'] + [p.relative_to(local).as_posix()
                    for p in (local / 'predictive_nesy').glob('*.py')]:
                ftp.put(str(local / relative), source + '/' + relative)
        script = f'''import json,os,pathlib,subprocess,time
root=pathlib.Path({ROOT!r});run=root/'pilot_planning_alignment';run.mkdir(exist_ok=True)
record=run/'launcher.json'
if record.exists():
    previous=json.loads(record.read_text());pid=previous['pid']
    try:
        command=pathlib.Path('/proc',str(pid),'cmdline').read_bytes()
    except FileNotFoundError:
        command=b''
    if b'pilot_planning.py' in command:
        raise RuntimeError('Previous pilot remains live; do not launch another')
if (run/'summary.json').exists():raise RuntimeError('Pilot already completed; inspect summary')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
cmd=[{PYTHON!r},'-u','pilot_planning.py','--checkpoint',str(root/'run_outcomes_v3/initial_memory_v2.pt'),'--out',str(run),'--updates','64']
log=(run/'driver.log').open('ab',buffering=0)
p=subprocess.Popen(cmd,cwd=root/'src_planning_pilot',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
data=dict(pid=p.pid,time=time.time(),device='cpu',command=cmd)
record.write_text(json.dumps(data,indent=2));print(json.dumps(data))
'''
        _, out, err = client.exec_command("python3 - <<'PLANNING_PILOT'\n" + script + "\nPLANNING_PILOT")
        print(out.read().decode() + err.read().decode())
        if out.channel.recv_exit_status():
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == '__main__':
    main()
