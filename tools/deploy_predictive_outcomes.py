"""Independent source snapshot and single-GPU outcome-supervised training."""
import argparse,hashlib,json,shlex
from pathlib import Path
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
local=Path(__file__).resolve().parents[1]/'predictive_rules'


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    c=connect()
    try:
        # A running experiment keeps its immutable source snapshot.
        _,out,err=c.exec_command('test -f '+ROOT+'/run_outcomes_v3/launcher.json && cat '+ROOT+'/run_outcomes_v3/launcher.json || true')
        old=out.read().decode().strip();out.channel.recv_exit_status()
        if old:raise RuntimeError('An outcome run was already launched; inspect it before replacing any source')
        files=sorted(x for x in local.rglob('*.py') if '__pycache__' not in x.parts and 'results' not in x.parts)
        dirs=sorted({ROOT+'/src_outcomes_v3/'+x.relative_to(local).parent.as_posix() for x in files})
        _,out,err=c.exec_command('mkdir -p '+' '.join(map(shlex.quote,dirs)))
        if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
        hashes={}
        with c.open_sftp() as ftp:
            for file in files:
                name=file.relative_to(local).as_posix();ftp.put(str(file),ROOT+'/src_outcomes_v3/'+name)
                hashes[name]=hashlib.sha256(file.read_bytes()).hexdigest()
            with ftp.open(ROOT+'/src_outcomes_v3/SOURCE_FILES.json','w') as stream:stream.write(json.dumps(hashes,indent=2))
        _,out,err=c.exec_command('cd '+ROOT+'/src_outcomes_v3 && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 '+PYTHON+' -m unittest test_contract test_memory test_counterfactual -v',timeout=180)
        result=out.read().decode()+err.read().decode();print(result,flush=True)
        (local/'outcome_contract_result.txt').write_text(result,encoding='utf-8')
        if out.channel.recv_exit_status():raise SystemExit(1)
        if not a.launch:return
        script=f'''import json,os,pathlib,subprocess,time,shutil
root=pathlib.Path({ROOT!r});run=root/'run_outcomes_v3';run.mkdir(exist_ok=True)
data=root/'counterfactual_training_v1';status=json.loads((data/'status.json').read_text())
if status['completed']<24:raise RuntimeError('Wait for at least 24 training trajectories before launching')
used=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>17500:raise RuntimeError('Preserving all processes: insufficient spare memory for this batch')
source=root/'run_memory_v2/best_development.pt';snapshot=run/'initial_memory_v2.pt'
if not snapshot.exists():shutil.copyfile(source,snapshot)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
log=(run/'driver.log').open('ab',buffering=0)
cmd=[{PYTHON!r},'train_memory.py','--out',str(run),'--updates','6000','--batch-size','16','--eval-every','400','--outcomes',str(data),'--cf-batch-size','8','--warm-start',str(snapshot),'--resume']
proc=subprocess.Popen(cmd,cwd=root/'src_outcomes_v3',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=proc.pid,time=time.time(),gpu=7,command=cmd,existing_processes_preserved=True)
(run/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
        _,out,err=c.exec_command("python3 - <<'OUTCOME_LAUNCH'\n"+script+"\nOUTCOME_LAUNCH",timeout=40)
        print(out.read().decode()+err.read().decode())
        if out.channel.recv_exit_status():raise SystemExit(1)
    finally:c.close()


if __name__=='__main__':main()
