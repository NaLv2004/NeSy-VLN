import json,os
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/v3')
name='checkpoint.e2.1791646884728491910.pt'
source=root/'runs/full_v3/latest'/name
target=root/'diagnostics/step2_checkpoint';target.mkdir(parents=True,exist_ok=True)
if not source.exists():
    print(json.dumps({'available':False,'source':str(source)}))
    raise SystemExit(0)
if not (target/name).exists():os.link(source,target/name)
marker={'checkpoint_file':name,'checkpoint_sha256':'b01a2417c906671c7ab2f3ff1b5715cc17179b23792f90ec3c359017104e7700',
        'epoch':2,'revision_interval':10,'horizon':32,'optimizer_included':True}
(target/'COMMITTED').write_text(json.dumps(marker,indent=2))
print(json.dumps({'available':True,'checkpoint':str(target),'hardlink':True}))
