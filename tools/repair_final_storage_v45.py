"""An exceptional lossless archive cycle after a verified storage stall."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');folder=base/'evidence_archives'
status=json.loads((folder/'active_maintenance_v38.json').read_text())
assert status['state']=='waiting' and status['next_cycle_not_before']-time.time()>180,status
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    if proc.parent.name==str(os.getpid()):continue
    try:assert (str(base/'archive_active_evidence.py')).encode() not in proc.read_bytes(),'Archive writer already active'
    except OSError:pass
waiting=list((base/'symbolic_execution_v38/runs').rglob('WAITING_FOR_STORAGE.json'))
assert waiting,'Storage exception no longer applies'
record={'time':time.time(),'verified_waits':{str(p):json.loads(p.read_text()) for p in waiting},
        'maintenance_next_cycle':status['next_cycle_not_before'],'before_free_gib':shutil.disk_usage(base).free/2**30}
completed=subprocess.run(['/usr/bin/python3',str(base/'archive_active_evidence.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
assert completed.returncode==0,completed.stdout[-4000:]
sys.path.insert(0,str(base/'symbolic_execution_v38'))
from nesy_vla.trace_images import collect_unreferenced_images
removed=collect_unreferenced_images(folder/'live_png_store')
record.update(finished=time.time(),free_gib=shutil.disk_usage(base).free/2**30,orphan_links_removed=len(removed),
              archive_result=json.loads((folder/'direct_archive_status.json').read_text()))
(base/'final_storage_repair_v45.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='archive_result'},indent=2))
