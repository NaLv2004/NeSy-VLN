import hashlib,json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_robust_v7')
assert not (root/'controller.json').exists(),'Inspect the existing controller, never launch twice'
left=json.loads((root/'configs/robust_reference.json').read_text());right=json.loads((root/'configs/robust_multieffect.json').read_text())
assert {k for k in left if left[k]!=right[k]}=={'architecture','program'}
assert {k for k in left['program'] if left['program'][k]!=right['program'][k]}=={'multi_effects'}
assert left['horizon']==32 and left['revision_interval']==10 and left['discrete_reasoning']
verification=json.loads((root/'training_verification.json').read_text());assert len(verification['arms'])==2
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['nesy_vla','scripts','tests','configs'] for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(root/'source_manifest.json').write_text(json.dumps({'time':time.time(),'files':files,'tests_passed':44},indent=2))
with (root/'controller.log').open('ab') as log:
    child=subprocess.Popen(['/usr/bin/python3','-u',str(root/'start_when_ready.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'time':time.time(),'pid':child.pid,'command':str(root/'start_when_ready.py'),'poll_interval_seconds':600}
(root/'controller.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
