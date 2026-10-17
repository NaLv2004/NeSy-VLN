import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_hybrid_replay_v54';OUT.mkdir(exist_ok=True)
contract=json.loads((OUT/'contract.json').read_text());assert contract['all_parameter_gradients_finite']
launch=OUT/'handoff.launcher.json'
if launch.exists():
    p=Path(f'/proc/{json.loads(launch.read_text())["pid"]}/cmdline')
    assert not p.exists() or not p.read_bytes(),'Handoff still live'
log=(OUT/'handoff.log').open('ab',buffering=0)
process=subprocess.Popen(['python3','-u',str(ROOT/'src_hybrid_replay_v54/handoff.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
row=dict(pid=process.pid,time=time.time());launch.write_text(json.dumps(row,indent=2));print(json.dumps(row))
