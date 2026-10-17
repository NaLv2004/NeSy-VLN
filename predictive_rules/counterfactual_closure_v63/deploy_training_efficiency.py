import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';SOURCE=ROOT+'/src_counterfactual_closure_v63'
c=connect()
try:
    with c.open_sftp() as ftp:
        for name in ('train_closure.py','closure_data.py'):
            ftp.put(str(HERE/name),SOURCE+'/'+name+'.efficient')
    code='''import json,os,signal,subprocess,time
from pathlib import Path
r=Path(%r);s=r/'src_counterfactual_closure_v63';o=r/'run_counterfactual_closure_v63'
v=json.loads((o/'supervisor.json').read_text());p=v['pid']
assert v['state']=='waiting_collection',v
assert b'continue_after_collection.py' in Path(f'/proc/{p}/cmdline').read_bytes()
fd=os.pidfd_open(p);signal.pidfd_send_signal(fd,signal.SIGSTOP)
try:
    fresh=json.loads((o/'supervisor.json').read_text());assert fresh['state']=='waiting_collection',fresh
    assert not (o/'latest.COMMITTED.json').exists()
    backup=o/('efficiency_'+str(int(time.time())));backup.mkdir()
    for name in ('train_closure.py','closure_data.py'):
        (backup/name).write_bytes((s/name).read_bytes());(s/(name+'.efficient')).replace(s/name)
    # The waiting supervisor already loaded its code. Its explicit --batch32
    # argument is read by the fitter; an on-disk config controls the override.
    (o/'efficiency.json').write_text(json.dumps(dict(batch_per_gpu=256,
        transition_holdout_batch=128,transition_holdout_interval=50,
        updates=2000,benchmark='counterfactual_closure_v63/throughput/training_benchmark.json',
        actual_optimizer_batch_increased=True,goal_navigation_evaluation_unchanged=True,
        time=time.time()),indent=2))
    subprocess.run(['python3','-m','compileall','-q',str(s)],check=True)
    print(json.dumps(dict(state='installed_before_training',supervisor=p,config=json.loads((o/'efficiency.json').read_text()))))
finally:
    signal.pidfd_send_signal(fd,signal.SIGCONT);os.close(fd)
'''%ROOT
    _,out,err=c.exec_command("python3 - <<'INSTALL_FIT_EFFICIENCY'\n"+code+'\nINSTALL_FIT_EFFICIENCY',timeout=120)
    print(out.read().decode());error=err.read().decode()
    if error:print(error)
    raise SystemExit(out.channel.recv_exit_status())
finally:c.close()
