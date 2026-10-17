"""Wait on existing process handles, then commit the completed cache once."""
import json,os,select,time
from pathlib import Path
from collections import Counter
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'replayed_boundary_features_v20'
launch=json.loads((out/'launcher.json').read_text());handles=[]
for item in launch['launchers']:
    try:handles.append(os.pidfd_open(item['pid']))
    except ProcessLookupError:pass
began=time.monotonic()
while handles:
    if time.monotonic()-began>900:raise TimeoutError('Existing replay process not complete; never restart automatically')
    ready,_,_=select.select(handles,[],[],50)
    for fd in ready:handles.remove(fd);os.close(fd)
    if handles:print(json.dumps(dict(waiting_for_existing_process_exit=True)),flush=True)
assert not (out/'manifest.json').exists()
records=[];totals=Counter();durations=[];scenes=None;vision=None
for shard in range(4):
    folder=out/f'shard_{shard}';status=json.loads((folder/'status.json').read_text());assert status['state']=='complete'
    report=json.loads((folder/'report.json').read_text());assert report['training_only']
    records+=report['reports'];totals.update(report['totals']);durations.append(report['seconds'])
    if scenes is None:scenes=report['train_layouts'];vision=report['frozen_vision']
    assert scenes==report['train_layouts'] and vision==report['frozen_vision']
assert len(set((r['bank'],r['episode']) for r in records))==len(records)
assert totals['states']==8066
# The input availability is a measured property, never a manufactured label.
assert totals['available_states']==totals['states'], totals
manifest=dict(training_only=True,records=records,totals=dict(totals),train_layouts=scenes,frozen_vision=vision,
    episodes=len(records),canonical_patch_agreement=totals['patch_matches']/totals['patches'],
    boundary_state_coverage=totals['available_states']/totals['states'],
    fixed_action_replay=True,contains_no_future_boundary_inputs=True,shard_seconds=durations,time=time.time())
temp=out/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2));os.replace(temp,out/'manifest.json')
print(json.dumps({k:v for k,v in manifest.items() if k not in {'records','frozen_vision'}}),flush=True)
