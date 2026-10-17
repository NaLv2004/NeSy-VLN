import json,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
handoff=json.loads((root/'adaptive_observed_v27_handoff.json').read_text())
assert handoff['state']=='launched'
oldcmd=Path('/proc/1202230/cmdline')
assert not oldcmd.exists() or not oldcmd.read_bytes()
path=root/'run_adaptive_observed_v26/status.json'
previous=json.loads(path.read_text())
previous.update(state='stopped_for_corrected_observation_training',stopped_time=time.time(),
                successor_run=str(root/'run_adaptive_observed_v27'),checkpoints_preserved=True,goal_complete=False)
temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(previous,indent=2));temporary.replace(path)
print(json.dumps(dict(old_run_state=previous['state'],old_final_update=previous['update'],new_pid=handoff['training']['pid'])))
