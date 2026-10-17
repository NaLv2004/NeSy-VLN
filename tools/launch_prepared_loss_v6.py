import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_loss_v6')
assert not (root/'controller.json').exists(),'Inspect existing controller instead of launching twice'
left=json.loads((root/'configs/loss_reference.json').read_text())
right=json.loads((root/'configs/loss_balanced.json').read_text())
assert {k for k in left if left[k]!=right[k]}=={'architecture','balanced_transition_loss'}
assert left['horizon']==32 and left['revision_interval']==10 and left['discrete_reasoning']
snapshot=root/'source_snapshot_initial';snapshot.mkdir()
files={}
for name in ['nesy_vla','scripts','configs','tests']:
    shutil.copytree(root/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
    for p in (snapshot/name).rglob('*'):
        if p.is_file():files[str(p.relative_to(snapshot))]=hashlib.sha256(p.read_bytes()).hexdigest()
(snapshot/'manifest.json').write_text(json.dumps({'time':time.time(),'files':files,'tests_passed':39},indent=2))
with (root/'controller.log').open('ab') as log:
    child=subprocess.Popen(['/usr/bin/python3','-u',str(root/'start_when_ready.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':child.pid,'time':time.time(),'command':str(root/'start_when_ready.py'),
    'poll_interval_seconds':600,'source_tests_passed':39}
(root/'controller.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
