"""Restore exact individual files from a content index; never overwrite unequal files."""
import argparse,hashlib,json,zlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('index');p.add_argument('--destination',required=True);p.add_argument('--suffix')
a=p.parse_args();root=Path(a.destination).resolve();record=json.loads(Path(a.index).read_text());count=0
for member in record['members']:
    if member.get('directory') or (a.suffix and not member['path'].endswith(a.suffix)):continue
    out=(root/member['path']).resolve();assert out.is_relative_to(root)
    data=zlib.decompress(Path(member['blob']).read_bytes())
    assert hashlib.sha256(data).hexdigest()==member['sha256'] and len(data)==member['bytes']
    if out.exists():assert out.read_bytes()==data
    else:out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    count+=1
print(json.dumps({'restored_files':count,'destination':str(root),'all_file_hashes_verified':True}))
