import json,time,shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_joint_v9')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def live(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
record=read(root/'integration_process.json');out=root/'integration'
result={'time':time.time(),'integration':record,'live':live(record['pid']),'status':read(out/'status.json'),
    'checkpoint':read(out/'latest/COMMITTED'),'free_gib':shutil.disk_usage(root).free/2**30}
path=out/'rank0.jsonl'
if path.exists():
    r=json.loads(path.read_text().splitlines()[-1]);result['training_record']={k:r[k] for k in ['update','episodes','successes','revision_events','symbolic_audit','gpu_peak_gib','losses','gradient_by_module']}
result['tail']=(root/'integration.log').read_text(errors='replace')[-1200:]
(root/'integration_inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
