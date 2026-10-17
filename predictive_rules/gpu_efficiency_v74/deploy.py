import ast,json,sys
from pathlib import Path
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
code='''
import json,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT);source=root/'src_gpu_efficiency_v74';out=root/'gpu_efficiency_v74';out.mkdir(exist_ok=True)
assert not (out/'launch.json').exists()
shutil.copytree(root/'src_continued_evidence_v72',source,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
'''.replace('Path(ROOT)','Path('+repr(ROOT)+')')
client=connect()
try:
    _,o,e=client.exec_command('python3 -',timeout=30)
    # Upload setup as a file to avoid shell quoting multiline code.
    with client.open_sftp() as ftp:
        with ftp.open(ROOT+'/setup_gpu_v74.py','w') as f:f.write(code)
    _,o,e=client.exec_command('python3 '+ROOT+'/setup_gpu_v74.py',timeout=30);assert o.channel.recv_exit_status()==0,e.read().decode()
    with client.open_sftp() as ftp:
        for p in HERE.glob('*.py'):
            if p.name=='deploy.py':continue
            ast.parse(p.read_text());ftp.put(str(p),ROOT+'/src_gpu_efficiency_v74/'+p.name)
        launch="""import json,subprocess,time
from pathlib import Path
root=Path(%r);out=root/'gpu_efficiency_v74'
with (out/'launcher.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['python3','-u',str(root/'src_gpu_efficiency_v74/after_exit.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,time=time.time());(out/'launch.json').write_text(json.dumps(r));print(json.dumps(r))
"""%ROOT
        with ftp.open(ROOT+'/launch_gpu_v74.py','w') as f:f.write(launch)
    _,o,e=client.exec_command('python3 '+ROOT+'/launch_gpu_v74.py',timeout=30);result=o.read().decode();assert o.channel.recv_exit_status()==0,e.read().decode()
    (HERE/'launch.json').write_text(result);print(result)
finally:client.close()
