import json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/frozen_vision_v8_code')
rows=subprocess.check_output(['ps','-eo','pid,etimes,pcpu,rss,args'],text=True).splitlines()
actual=[row for row in rows if 'python -u scripts/verify_pretrained_vision.py' in row and ' -c ' not in row]
logs={name:(root/name).read_text(errors='replace')[-1800:] if (root/name).exists() else None for name in ['verification.log','pretrained_vision_verification.log']}
print(json.dumps({'time':time.time(),'actual_processes':actual,'logs':logs},indent=2))
