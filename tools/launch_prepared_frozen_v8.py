import hashlib,json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_vision_v8')
assert not (root/'controller.json').exists()
left=json.loads((root/'configs/frozen_convnext.json').read_text());right=json.loads((root/'configs/frozen_qwen.json').read_text())
assert {k for k in left if left[k]!=right[k]}=={'architecture','vision_backbone'}
assert left['discrete_reasoning'] and left['horizon']==32 and left['revision_interval']==10
manifest=json.loads((root/'experiment.json').read_text())
manifest['source_files']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['nesy_vla','scripts','tests','configs'] for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(root/'experiment.json').write_text(json.dumps(manifest,indent=2))
with (root/'controller.log').open('ab') as log:
    child=subprocess.Popen(['/usr/bin/python3','-u',str(root/'start_when_ready.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':child.pid,'time':time.time(),'command':str(root/'start_when_ready.py'),'poll_interval_seconds':600}
(root/'controller.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
