import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'paired_goal_probe_v12';out.mkdir(exist_ok=True)
marker=out/'controller_process.json'
if marker.exists():
    old=json.loads(marker.read_text());p=Path(f"/proc/{old['pid']}/cmdline")
    if p.exists() and b'start_pair_probe_when_ready.py' in p.read_bytes():print(json.dumps({'already_live':old}));raise SystemExit
with (out/'controller.log').open('ab') as log:
    process=subprocess.Popen(['python3','-u',str(base/'start_pair_probe_when_ready.py')],
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':process.pid,'time':time.time(),'controller':str(base/'start_pair_probe_when_ready.py')}
marker.write_text(json.dumps(record,indent=2));print(json.dumps(record))
