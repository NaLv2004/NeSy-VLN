import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
SOURCE=ROOT+'/src_counterfactual_closure_v63'
c=connect()
try:
    with c.open_sftp() as ftp:
        ftp.put(str(HERE/'collect_closure.py'),SOURCE+'/collect_closure.py.next')
        ftp.put(str(HERE/'batch_future.py'),SOURCE+'/batch_future.py')
    code=(HERE/'install_batched_collection.py').read_text(encoding='utf-8')
    _,out,err=c.exec_command("python3 -u - <<'INSTALL_BATCH_ENCODING'\n"+code+'\nINSTALL_BATCH_ENCODING',timeout=600)
    for line in out:print(line,end='',flush=True)
    error=err.read().decode()
    if error:print(error)
    raise SystemExit(out.channel.recv_exit_status())
finally:c.close()
