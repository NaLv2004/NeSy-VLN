import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'tools'))
from predictive_remote import connect
SOURCE='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_counterfactual_closure_v63'
client=connect()
try:
    _,out,err=client.exec_command('mkdir -p '+SOURCE);assert out.channel.recv_exit_status()==0,err.read().decode()
    with client.open_sftp() as ftp:
        for name in ('collect_closure.py','closure_worker.py','model.py','planner.py','data.py','evaluate.py','program_archive.py','coordinate_collection.py'):
            ftp.put(str(HERE/name),SOURCE+'/'+name)
    code=(HERE/'launch_collection.py').read_text(encoding='utf-8')
    _,out,err=client.exec_command("python3 - <<'LAUNCH_CLOSURE'\n"+code+"\nLAUNCH_CLOSURE",timeout=120)
    print(out.read().decode());error=err.read().decode()
    if error:print(error)
    raise SystemExit(out.channel.recv_exit_status())
finally:client.close()
