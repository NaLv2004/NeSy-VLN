import json,os,shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010');processes=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
    if int(path.parent.name)==os.getpid():continue
    try:
        raw=path.read_bytes();args=raw.split(b'\0')
        if args and Path(args[0].decode()).name=='python3' and b'Losslessly archive only bulky trace evidence' in raw:
            processes.append({'pid':int(path.parent.name),'stat':(path.parent/'stat').read_text()})
    except (OSError,UnicodeError):pass
files=[{'name':p.name,'bytes':p.stat().st_size} for p in (root/'evidence_archives').glob('*') if p.is_file()]
print(json.dumps({'processes':processes,'files':files,'free_bytes':shutil.disk_usage(root).free},indent=2))
