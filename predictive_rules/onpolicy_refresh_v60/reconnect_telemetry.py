"""Restart only the read-only telemetry child; local server reconnects itself."""
import json,os,signal,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
target=str(ROOT/'dashboard/remote_stream.py');killed=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:parts=(proc/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError):continue
    if target.encode() in parts and any(b'python' in x for x in parts[:1]):
        os.kill(int(proc.name),signal.SIGTERM);killed.append(int(proc.name))
pid=2186519;p=Path(f'/proc/{pid}/cmdline')
record=dict(time=time.time(),telemetry_children_restarted=killed,
    handoff=json.loads((ROOT/'diagnostic_onpolicy_refresh_v60/handoff.status.json').read_text()),
    handoff_command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else '')
(ROOT/'diagnostic_onpolicy_refresh_v60/handoff_verified.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
