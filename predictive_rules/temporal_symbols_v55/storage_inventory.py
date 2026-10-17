import json,os,shutil
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
files=[];directories=[]
for folder in ROOT.iterdir():
    if not folder.is_dir():continue
    size=0
    for base,ds,fs in os.walk(folder):
        ds[:]=[d for d in ds if d!='__pycache__']
        for name in fs:
            p=Path(base)/name
            try:st=p.stat()
            except OSError:continue
            size+=st.st_size
            if st.st_size>100_000_000 and ('backward' in name or 'case' in name or 'replay' in name or 'cache' in name):files.append(dict(path=str(p),bytes=st.st_size))
    directories.append(dict(path=folder.name,bytes=size))
print(json.dumps(dict(free=shutil.disk_usage(ROOT).free,debug_files=files,largest=sorted(directories,key=lambda v:-v['bytes'])[:15])))
