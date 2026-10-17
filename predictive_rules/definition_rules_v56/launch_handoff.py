"""Dispatch a single verified event-driven handoff, not a poll loop."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
DIAG=ROOT/'diagnostic_definition_rules_v56';SOURCE=ROOT/'src_definition_rules_v56'
assert not (DIAG/'handoff.launch.json').exists(),'Already dispatched; inspect saved PID'
contract=json.loads((DIAG/'contract.json').read_text())
assert contract['real_optimizer_step'] and contract['pure_numpy_planner_verified']
with (DIAG/'handoff.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(SOURCE/'handoff.py')],cwd=SOURCE,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,time=time.time(),source=str(SOURCE),predecessor='run_temporal_symbols_v55')
(DIAG/'handoff.launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
