import hashlib,json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/symbolic_paired_v12/integration')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
process=read(out/'process.json');p=Path(f"/proc/{process['pid']}/cmdline");commit=read(out/'latest/COMMITTED')
rows=(out/'rank0.jsonl').read_text().splitlines() if (out/'rank0.jsonl').exists() else []
row=json.loads(rows[-1]) if rows else None
record={'time':time.time(),'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
    'status':read(out/'status.json'),'commit':commit,'tail':(out/'integration.log').read_text(errors='replace')[-1500:],
    'last_update':row,'commit_sha_verified':bool(commit and hashlib.sha256((out/'latest'/commit['checkpoint_file']).read_bytes()).hexdigest()==commit['checkpoint_sha256'])}
(out/'verification.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='last_update'},indent=2))
if row:print(json.dumps({k:row.get(k) for k in ['demonstration_episode_ids','symbolic_audit','gradient_by_module','gpu_peak_gib','losses']},indent=2))
