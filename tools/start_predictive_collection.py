"""Start or resume the isolated training-label collector, preserving live jobs."""
import json,shlex
from pathlib import Path
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
local=Path(__file__).resolve().parents[1]/'predictive_rules'
c=connect()
try:
    with c.open_sftp() as ftp:
        for name in ['collect_counterfactuals.py','counterfactual_worker.py','diagnose_saved.py','audit_trace.py']:
            ftp.put(str(local/name),ROOT+'/src_memory_v2/'+name)
    script=f'''import json,os,pathlib,subprocess,time
root=pathlib.Path({ROOT!r});out=root/'counterfactual_training_v1';out.mkdir(exist_ok=True)
if (out/'launcher.json').exists():
 old=json.loads((out/'launcher.json').read_text());stat=pathlib.Path('/proc')/str(old['pid'])/'stat'
 if stat.exists() and stat.read_text().split(') ',1)[1].split()[0]!='Z':
  print(json.dumps(dict(already_running=old)));raise SystemExit(0)
used=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>19000:raise RuntimeError('Insufficient spare GPU7 memory; no process stopped')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
log=(out/'driver.log').open('ab',buffering=0)
proc=subprocess.Popen([{PYTHON!r},'collect_counterfactuals.py','--out',str(out),'--episodes','96'],cwd=root/'src_memory_v2',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=proc.pid,time=time.time(),gpu=7,command=proc.args)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
    _,out,err=c.exec_command("python3 - <<'COLLECT_LAUNCH'\n"+script+"\nCOLLECT_LAUNCH",timeout=30)
    print(out.read().decode()+err.read().decode())
    if out.channel.recv_exit_status():raise SystemExit(1)
    command='cd '+ROOT+'/src_memory_v2 && '+PYTHON+' audit_trace.py '+ROOT+'/run_memory_v2/development_step_000800 && '+PYTHON+' diagnose_saved.py '+ROOT+'/run_memory_v2/development_step_000800'
    _,out,err=c.exec_command(command,timeout=180)
    print(out.read().decode()+err.read().decode())
    if out.channel.recv_exit_status():raise SystemExit(1)
finally:c.close()
