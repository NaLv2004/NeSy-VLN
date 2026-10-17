"""Upload only this experiment and run its tests on the remote host."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
from predictive_remote import connect

LOCAL = Path(__file__).resolve().parents[1] / 'predictive_rules'
REMOTE = '/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
PYTHON = '/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    client=connect()
    try:
        files=sorted(p for p in LOCAL.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md') and '__pycache__' not in p.parts and p.name!='inspect_remote.py')
        paths=sorted({REMOTE+'/src/'+p.relative_to(LOCAL).parent.as_posix() for p in files})
        _,out,err=client.exec_command('mkdir -p '+' '.join(shlex.quote(p) for p in paths))
        if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
        manifest={}
        with client.open_sftp() as ftp:
            for path in files:
                relative=path.relative_to(LOCAL).as_posix()
                ftp.put(str(path),REMOTE+'/src/'+relative)
                manifest[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
            with ftp.open(REMOTE+'/src/SOURCE_FILES.json','w') as stream:
                stream.write(json.dumps(manifest,indent=2))
        command='cd '+shlex.quote(REMOTE+'/src')+' && CUDA_VISIBLE_DEVICES=7 '+shlex.quote(PYTHON)+' -m unittest test_contract -v'
        _,out,err=client.exec_command(command,timeout=180)
        output=out.read().decode()+err.read().decode()
        code=out.channel.recv_exit_status()
        print(output,flush=True)
        (LOCAL/'contract_test_result.txt').write_text(output,encoding='utf-8')
        if code:raise SystemExit(code)
        if args.launch:
            script=f'''import os,json,subprocess,time,pathlib
root=pathlib.Path({REMOTE!r});run=root/'run_v1';run.mkdir(exist_ok=True)
info=subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
if int(info.strip())>500:raise RuntimeError('GPU 7 is no longer free; existing processes are preserved')
if (run/'status.json').exists():
 s=json.loads((run/'status.json').read_text())
 if pathlib.Path('/proc/'+str(s.get('pid',-1))).exists():raise RuntimeError('Existing job is still live')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
log=(run/'driver.log').open('ab',buffering=0)
process=subprocess.Popen([{PYTHON!r},'train.py','--out',str(run),'--updates','6000','--batch-size','16','--eval-every','200','--checkpoint-every','100','--resume'],cwd=root/'src',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={{'pid':process.pid,'gpu':7,'time':time.time(),'directory':str(run),'command':process.args}}
(run/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
            _,out,err=client.exec_command("python3 - <<'LAUNCH_PREDICTIVE'\n"+script+"\nLAUNCH_PREDICTIVE",timeout=30)
            print(out.read().decode()+err.read().decode())
            if out.channel.recv_exit_status():raise SystemExit(1)
    finally:
        client.close()


if __name__=='__main__':main()
