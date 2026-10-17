import json,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_robust_v7'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def process(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
record={'time':time.time(),'controller':read(root/'controller.json'),'allocation':read(root/'allocation.json'),'runs':{}}
record['controller_live']=process(record['controller']['pid'])
for name in ['robust_reference','robust_multieffect']:
    out=root/'runs'/name;supervisor=read(out/'supervisor.json')
    data={'supervisor':supervisor,'status':read(out/'status.json'),'checkpoint':read(out/'latest/COMMITTED')}
    if supervisor:data['training_live']=process(supervisor.get('pid'))
    rank=out/'rank0.jsonl'
    if rank.exists():
        rows=rank.read_text().splitlines();last=json.loads(rows[-1]);data['last']={k:last[k] for k in ['update','seconds','gpu_peak_gib','training_frames','losses']}
    logs=sorted(out.glob('launch*.log'),key=lambda p:p.stat().st_mtime)
    if not data['status'] and logs:data['startup_tail']=logs[-1].read_text(errors='replace')[-1600:]
    record['runs'][name]=data
record['controller_tail']=(root/'controller.log').read_text(errors='replace')[-2000:]
record['archive']=read(base/'evidence_archives/v4_controller.json')
record['archive_live']=process(record['archive']['pid']) if record['archive'] else None
record['archive_tail']=(base/'evidence_archives/v4_archive.log').read_text(errors='replace')[-1500:]
record['free_bytes']=shutil.disk_usage(base).free
(root/'startup_inspection.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
