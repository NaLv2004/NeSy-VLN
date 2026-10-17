import json,os
from pathlib import Path
roots=[Path('/home/user/xjy/strips_wm_navigation_20261009/runs'),Path('/home/user/xjy/nesy_vla_20261010')]
live=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        s=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if 'python' in s and any(str(r) in s for r in roots):live.append(s)
    except (OSError,PermissionError):pass
out=[]
for root in roots:
    for p in root.rglob('*.pt'):
        if p.is_symlink() or not any('smoke' in x or x=='integration' for x in p.relative_to(root).parts):continue
        if p.stat().st_nlink>1:continue
        run=next((a for a in p.parents if 'smoke' in a.name or a.name=='integration'),None)
        if any(str(run) in cmd for cmd in live):continue
        out.append({'path':str(p),'bytes':p.stat().st_size,'run':str(run)})
result={'candidates':out,'total':sum(r['bytes'] for r in out),'live_commands':live}
dest=roots[1]/'retired_smoke_cleanup_candidates.json';dest.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
