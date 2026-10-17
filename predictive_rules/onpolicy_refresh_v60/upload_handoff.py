import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
SOURCE=ROOT+'/src_onpolicy_refresh_v60_training';DIAG=ROOT+'/diagnostic_onpolicy_refresh_v60'
files=['model.py','definition_rule.py','planner.py','program_archive.py','evaluate.py','data.py','objectives.py',
       'temporal_objective.py','goal_objective.py','train.py','supervisor.py','storage.py','check_contract.py','handoff.py']
c=connect()
try:
    _,out,err=c.exec_command('mkdir -p '+SOURCE);assert out.channel.recv_exit_status()==0
    with c.open_sftp() as ftp:
        for name in files:ftp.put(str(HERE/name),SOURCE+'/'+name)
    code="""import json,os,shutil,subprocess,time
from pathlib import Path
root=Path(%r);source=Path(%r);diag=Path(%r)
if not (source/'predictive_nesy').exists():shutil.copytree(root/'src_goal_equivalence_v57/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
subprocess.run(['python3','-m','compileall','-q',str(source)],check=True)
assert not (diag/'handoff.launch.json').exists(),'Inspect existing handoff before any retry'
log=(diag/'handoff.log').open('ab',buffering=0)
p=subprocess.Popen(['python3','-u',str(source/'handoff.py')],cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,time=time.time(),source=str(source),collector_pid=2184984)
(diag/'handoff.launch.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
"""%(ROOT,SOURCE,DIAG)
    _,out,err=c.exec_command("python3 - <<'HANDOFF_PREPARE'\n"+code+"\nHANDOFF_PREPARE",timeout=120)
    print(out.read().decode());error=err.read().decode()
    if error:print(error)
    raise SystemExit(out.channel.recv_exit_status())
finally:c.close()
