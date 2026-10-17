import json,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'evidence_archives';marker=out/'v5_spatial_controller.json'
assert not marker.exists(),'Inspect existing archive controller before restarting'
old=base/'symbolic_sft_v5/runs/sft_spatial';assert (old/'STOP').exists()
script=base/'robust_grounding_v7_code/archive_evidence.py'
command=['/usr/bin/python3','-u',str(script),'--runs','symbolic_sft_v5/runs/sft_spatial']
with (out/'v5_spatial_archive.log').open('ab') as log:
    process=subprocess.Popen(command,cwd=base,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'time':time.time(),'command':command};marker.write_text(json.dumps(record,indent=2));print(json.dumps(record))
