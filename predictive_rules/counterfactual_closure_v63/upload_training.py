"""Install the isolated fitter and a handle-based collection-to-fit handoff."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';SOURCE=ROOT+'/src_counterfactual_closure_v63'
c=connect()
try:
    with c.open_sftp() as ftp:
        for name in ('closure_objective.py','closure_data.py','train_closure.py','check_objective.py','continue_after_collection.py'):
            ftp.put(str(HERE/name),SOURCE+'/'+name)
    code="""import json,os,subprocess,time
from pathlib import Path
root=Path(%r);source=Path(%r);out=root/'run_counterfactual_closure_v63';data=root/'counterfactual_closure_v63'
out.mkdir(exist_ok=True);assert not (out/'launcher.json').exists(),'Inspect prior handoff, never duplicate it'
state=json.loads((data/'status.json').read_text());assert state['state'] in ('collecting','complete'),state
pid=json.loads((data/'launcher.json').read_text())['pid']
p=Path(f'/proc/{pid}/cmdline')
if state['state']=='collecting':assert p.exists() and b'coordinate_collection.py' in p.read_bytes()
subprocess.run(['python3','-m','compileall','-q',str(source)],check=True)
log=(out/'supervisor.log').open('ab',buffering=0)
child=subprocess.Popen(['python3','-u',str(source/'continue_after_collection.py')],cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,time=time.time(),wait_for_collection_pid=pid,updates=2000,gpus=[6,7],
    full80_evaluations_after_fitting=1,finalfixed80_used=False,training_started=False)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
"""%(ROOT,SOURCE)
    _,out,err=c.exec_command("python3 - <<'INSTALL_CLOSURE_FITTER'\n"+code+"\nINSTALL_CLOSURE_FITTER",timeout=120)
    print(out.read().decode());error=err.read().decode()
    if error:print(error)
    raise SystemExit(out.channel.recv_exit_status())
finally:c.close()
