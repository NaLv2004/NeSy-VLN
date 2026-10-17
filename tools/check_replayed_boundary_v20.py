import json,os,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/replayed_boundary_features_v20')
launch=json.loads((root/'launcher.json').read_text());reports=[]
for row in launch['launchers']:
    process=Path('/proc')/str(row['pid']);alive=process.exists()
    cmd=(process/'cmdline').read_bytes().decode().replace('\0',' ') if alive else ''
    path=root/f"shard_{row['shard']}"/'status.json'
    reports.append(dict(pid=row['pid'],gpu=row['gpu'],alive=alive,identity_verified=alive and 'recover_boundary_features_v20.py' in cmd,
        status=json.loads(path.read_text()) if path.exists() else None,report_complete=(path.parent/'report.json').exists(),
        driver_tail=(root/f"driver_{row['shard']}.log").read_text()[-600:]))
print(json.dumps(dict(time=time.time(),shards=reports)))
