import hashlib,json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_joint_v9')
assert not (root/'controller.json').exists()
record=json.loads((root/'integration_canonical_process.json').read_text())
assert not Path(f"/proc/{record['pid']}/cmdline").exists()
assert json.loads((root/'integration_canonical/status.json').read_text())['state']=='complete'
row=json.loads((root/'integration_canonical/rank0.jsonl').read_text().splitlines()[-1])
assert row['losses']['ppo_ratio_max_error']==0 and row['losses']['on_policy_fresh_bit_disagreement_metric']==0
manifest=json.loads((root/'experiment.json').read_text())
manifest['integration_verification']={k:row[k] for k in ['episodes','revision_events','losses','symbolic_audit','gpu_peak_gib']}
manifest['source_files']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['nesy_vla','scripts','tests','configs'] for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(root/'experiment.json').write_text(json.dumps(manifest,indent=2))
with (root/'controller.log').open('ab') as log:
    child=subprocess.Popen(['/usr/bin/python3','-u',str(root/'start_when_ready.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
result={'pid':child.pid,'time':time.time(),'command':str(root/'start_when_ready.py'),'poll_interval_seconds':600}
(root/'controller.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
