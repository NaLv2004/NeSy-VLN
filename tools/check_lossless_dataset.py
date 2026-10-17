import json,time,shutil
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/data/lossless_demonstrations_v7');saved=out/'inspection.json'
def read(path):return json.loads(path.read_text()) if path.exists() else None
previous=read(saved)
if previous and previous.get('healthy') and not previous.get('ready') and time.time()-previous['time']<600:
    print(json.dumps({'skipped':'long task minimum600 seconds','last_time':previous['time']}));raise SystemExit(0)
controller=read(out/'controller.json');jobs=[]
for entry in (controller or {}).get('jobs',[]):
    pid=entry['pid'];proc=Path(f'/proc/{pid}/cmdline')
    jobs.append(dict(entry,live_command=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
        progress=read(out/f'progress_{entry["shard"]}.json'),completed=(out/f'shard_{entry["shard"]}.json').exists()))
ready=read(out/'READY.json')
result={'time':time.time(),'controller':controller,'jobs':jobs,'ready':ready,
    'healthy':bool(jobs) and all(j['live_command'] or j['completed'] for j in jobs),'free_bytes':shutil.disk_usage(out).free}
if not result['healthy']:result['error']=(out/'controller.log').read_text(errors='replace')[-3000:]
saved.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
