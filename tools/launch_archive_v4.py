import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010');out=root/'evidence_archives'
marker=out/'v4_controller.json'
if marker.exists():raise RuntimeError('Inspect existing archive process before restarting')
script=root/'robust_grounding_v7_code/archive_evidence.py'
command=['/usr/bin/python3','-u',str(script),'--runs','discrete_v4/runs/discrete_gru','discrete_v4/runs/discrete_vlm']
with (out/'v4_archive.log').open('ab') as log:
    child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':child.pid,'time':time.time(),'command':command};marker.write_text(json.dumps(record,indent=2));print(json.dumps(record))
