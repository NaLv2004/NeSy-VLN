import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
REMOTE='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_onpolicy_refresh_v60'
c=connect()
try:
    with c.open_sftp() as ftp:
        for p in HERE.glob('*.py'):ftp.put(str(p),REMOTE+'/'+p.name)
    _,out,err=c.exec_command('python3 -m compileall -q '+REMOTE+' && python3 '+REMOTE+'/launch_collection.py',timeout=120)
    print(out.read().decode());error=err.read().decode()
    if error:print(error)
    raise SystemExit(out.channel.recv_exit_status())
finally:c.close()
