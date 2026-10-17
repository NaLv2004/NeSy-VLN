import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'lossless_dataset_v7_code';out=base/'data/lossless_demonstrations_v7'
out.mkdir(parents=True,exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect the existing controller before attempting a resume'
with (out/'controller.log').open('ab') as log:
    process=subprocess.Popen(['/usr/bin/python3','-u',str(root/'control_lossless_dataset.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'time':time.time(),'command':str(root/'control_lossless_dataset.py')}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
