from pathlib import Path
import json
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
matches=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    if ('prepare_entmax_reference.py' in command and 'python' in command) or ('pip install' in command and str(ROOT/'vendor_entmax_1_3') in command):
        matches.append(dict(pid=int(proc.name),command=command))
print(json.dumps(dict(installer_processes=matches,installed=(ROOT/'vendor_entmax_1_3/entmax/__init__.py').is_file())))
