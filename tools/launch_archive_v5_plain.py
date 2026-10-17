import json,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'evidence_archives';marker=out/'v5_plain_controller.json'
assert not marker.exists(),'Inspect existing archive controller before restarting'
old=base/'symbolic_sft_v5/runs/sft_plain';assert (old/'STOP').exists()
command=['/usr/bin/python3','-u',str(base/'robust_grounding_v7_code/archive_evidence.py'),'--runs','symbolic_sft_v5/runs/sft_plain']
with (out/'v5_plain_archive.log').open('ab') as log:
    process=subprocess.Popen(command,cwd=base,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'time':time.time(),'command':command};marker.write_text(json.dumps(record,indent=2));print(json.dumps(record))
