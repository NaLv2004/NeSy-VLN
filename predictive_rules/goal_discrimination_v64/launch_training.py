import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';SOURCE=ROOT+'/src_goal_discrimination_v64'
c=connect()
try:
    with c.open_sftp() as ftp:
        for name in ('model.py','objectives.py','data.py','evaluate.py','planner.py','program_archive.py',
                'temporal_objective.py','goal_objective.py','accelerated_transition.py','feedback_objective.py',
                'batch_vision.py','train_feedback.py','size_training.py','supervise_training.py'):
            ftp.put(str(HERE/name),SOURCE+'/'+name)
        for name in ('status.json','goal_alias_audit.json','manifest.json'):
            ftp.get(ROOT+'/goal_discrimination_v64/'+name,str(HERE/('collection_'+name if name!='goal_alias_audit.json' else name)))
    code='''import json,os,subprocess,time
from pathlib import Path
r=Path(%r);s=Path(%r);data=r/'goal_discrimination_v64';out=r/'run_goal_discrimination_v64';out.mkdir(exist_ok=True)
status=json.loads((data/'status.json').read_text());assert status['state']=='complete'
for pid in [status['pid'],*status.get('workers',[])]:
    p=Path(f'/proc/{pid}/cmdline');assert not p.exists() or not p.read_bytes(),(pid,p.read_bytes())
assert not (out/'launcher.json').exists(),'Inspect existing training; never duplicate'
assert not subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPUs6/7occupied'
subprocess.run(['python3','-m','compileall','-q',str(s)],check=True)
with (out/'supervisor.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(s/'supervise_training.py')],cwd=s,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),gpus=[6,7],updates=1000,finalfixed80_used=False,
    collection=status,no_other_processes_stopped=True)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''%(ROOT,SOURCE)
    _,o,e=c.exec_command("python3 - <<'LAUNCH_TRAINING_V64'\n"+code+'\nLAUNCH_TRAINING_V64',timeout=120)
    answer=o.read().decode();print(answer);error=e.read().decode()
    if error:print(error)
    rc=o.channel.recv_exit_status()
    if rc==0:(HERE/'training_launch.json').write_text(answer,encoding='utf-8')
    raise SystemExit(rc)
finally:c.close()
