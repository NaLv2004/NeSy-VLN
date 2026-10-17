import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_temporal_symbols_v55';SOURCE=ROOT/'src_temporal_symbols_v55'
assert json.loads((OUT/'contract.json').read_text())['real_optimizer_step']
path=OUT/'handoff.launcher.json'
if path.exists():
    previous=json.loads(path.read_text());cmd=Path(f'/proc/{previous["pid"]}/cmdline')
    assert not cmd.exists() or not cmd.read_bytes(),'Existing handoff still live'
    assert json.loads((OUT/'handoff.status.json').read_text())['state']!='training_dispatched'
with (OUT/'handoff.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(SOURCE/'handoff.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
row=dict(pid=child.pid,time=time.time());path.write_text(json.dumps(row,indent=2));print(json.dumps(row))
