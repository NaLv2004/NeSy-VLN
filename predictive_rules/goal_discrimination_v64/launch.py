import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';SOURCE=ROOT+'/src_goal_discrimination_v64'
c=connect()
try:
    _,o,e=c.exec_command('mkdir -p '+SOURCE);assert o.channel.recv_exit_status()==0,e.read().decode()
    with c.open_sftp() as ftp:
        for name in ('collect_feedback.py','replay_worker.py','coordinate_collection.py'):
            ftp.put(str(HERE/name),SOURCE+'/'+name)
    code='''import json,os,subprocess,time
from pathlib import Path
r=Path(%r);s=Path(%r);out=r/'goal_discrimination_v64';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect prior launch; never duplicate'
assert json.loads((r/'run_counterfactual_closure_v63/supervisor.json').read_text())['state']=='finished'
assert not subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
best=json.loads((r/'run_counterfactual_closure_v63/best.COMMITTED.json').read_text())
os.link(r/'run_counterfactual_closure_v63'/best['filename'],out/'source.pt')
(out/'source.COMMITTED.json').write_text(json.dumps(best,indent=2))
data=json.loads((r/'counterfactual_closure_v63/manifest.json').read_text())
manifest=dict(training_only=True,source=best,episode_ids=data['episode_ids'],train_layouts=data['train_layouts'],
    independent_finalfixed80_used=False,dev80_used=False,max_actions=32,existing_symbolic_actions_replayed=True,
    label='Existing environment STOP success indicator; training supervision only',
    no_runtime_feedback_input=True,no_hand_assigned_state_ids=True,no_map=True,
    no_coordinates_or_distances_or_collision_labels=True,batch_image_encoding=256,simulators_per_gpu=24)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
subprocess.run(['python3','-m','compileall','-q',str(s)],check=True)
with (out/'coordinator.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(s/'coordinate_collection.py')],cwd=s,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),gpus=[6,7],tasks=600,train_only=True)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''%(ROOT,SOURCE)
    _,o,e=c.exec_command("python3 - <<'LAUNCH_GOAL_FEEDBACK'\n"+code+'\nLAUNCH_GOAL_FEEDBACK',timeout=120)
    print(o.read().decode());error=e.read().decode()
    if error:print(error)
    raise SystemExit(o.channel.recv_exit_status())
finally:c.close()
