import json,sys,torch
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');reference=Path('/home/user/xjy/strips_wm_navigation_20261009')
sys.path.insert(0,str(reference));from stripsnav.io import load_committed
inventory=json.loads((base/'disk_inventory_20261011.json').read_text());entries=[];seen=set()
for row in inventory['large_files']:
    path=Path(row['path']);folder=path.parent
    if not path.is_relative_to(reference/'runs') or folder in seen or not (folder/'COMMITTED').exists():continue
    seen.add(folder)
    marker=json.loads((folder/'COMMITTED').read_text())
    if 'checkpoint_sha256' not in marker:continue
    entries.append({'folder':str(folder),'bytes':row['bytes'],'marker':marker})
result={'candidates':entries,'samples':[]}
for row in entries[:5]:
    data=load_committed(row['folder']);result['samples'].append({'folder':row['folder'],
        'keys':{key:type(value).__name__ for key,value in data.items()}});del data
(base/'cold_checkpoint_payload_inventory.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'count':len(entries),'samples':result['samples'],'first_folders':[r['folder'] for r in entries[:30]]},indent=2))
