"""Deploy v2 separately; share only the already assigned GPU 7, kill nothing."""
import argparse,json,hashlib,shlex
from pathlib import Path
from predictive_remote import connect

LOCAL=Path(__file__).resolve().parents[1]/'predictive_rules'
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    client=connect()
    try:
        files=sorted(x for x in LOCAL.rglob('*.py') if '__pycache__' not in x.parts and 'results' not in x.parts)
        directories=sorted({ROOT+'/src_memory_v2/'+x.relative_to(LOCAL).parent.as_posix() for x in files})
        _,out,err=client.exec_command('mkdir -p '+' '.join(map(shlex.quote,directories)))
        if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
        hashes={}
        with client.open_sftp() as ftp:
            for file in files:
                name=file.relative_to(LOCAL).as_posix();ftp.put(str(file),ROOT+'/src_memory_v2/'+name)
                hashes[name]=hashlib.sha256(file.read_bytes()).hexdigest()
            with ftp.open(ROOT+'/src_memory_v2/SOURCE_FILES.json','w') as stream:stream.write(json.dumps(hashes,indent=2))
        _,out,err=client.exec_command('cd '+ROOT+'/src_memory_v2 && CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 '+PYTHON+' -m unittest test_contract test_memory -v',timeout=180)
        text=out.read().decode()+err.read().decode();print(text,flush=True)
        (LOCAL/'memory_contract_result.txt').write_text(text,encoding='utf-8')
        if out.channel.recv_exit_status():raise SystemExit(1)
        if not a.launch:return
        script=f'''import json,os,pathlib,subprocess,time
root=pathlib.Path({ROOT!r});run=root/'run_memory_v2';run.mkdir(exist_ok=True)
def owned(pid):
 for _ in range(8):
  proc=pathlib.Path('/proc')/str(pid)
  if not proc.exists():return False
  try:
   cwd=str((proc/'cwd').resolve());text=(proc/'stat').read_text();rest=text[text.rfind(')')+2:].split()
   if cwd.startswith(str(root)+'/'):return True
   pid=int(rest[1])
  except (OSError,ValueError):return False
 return False
uuid=subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
for row in processes.splitlines():
 fields=[x.strip() for x in row.split(',')]
 if fields[0]==uuid and not owned(int(fields[1])):raise RuntimeError('GPU7 contains an unrelated process; preserved without launching')
used=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>6000:raise RuntimeError('Not enough measured free memory for the new batch; no process stopped')
if (run/'launcher.json').exists():
 pid=json.loads((run/'launcher.json').read_text())['pid'];stat=pathlib.Path('/proc')/str(pid)/'stat'
 if stat.exists() and stat.read_text().split(') ',1)[1].split()[0]!='Z':raise RuntimeError('v2 is already live')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
log=(run/'driver.log').open('ab',buffering=0)
process=subprocess.Popen([{PYTHON!r},'train_memory.py','--out',str(run),'--updates','6000','--batch-size','64','--eval-every','400','--resume'],cwd=root/'src_memory_v2',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={{'pid':process.pid,'gpu':7,'time':time.time(),'directory':str(run),'command':process.args,'existing_processes_preserved':True}}
(run/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
        _,out,err=client.exec_command("python3 - <<'LAUNCH_MEMORY'\n"+script+"\nLAUNCH_MEMORY",timeout=30)
        print(out.read().decode()+err.read().decode())
        if out.channel.recv_exit_status():raise SystemExit(1)
    finally:client.close()


if __name__=='__main__':main()
