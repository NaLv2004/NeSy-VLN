"""Launch the independently grounded-memory experiment after v41 drains."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=ROOT/'src_grounded_memory_v42';out=ROOT/'run_grounded_memory_v42';old=ROOT/'run_stable_onpolicy_v41'

def command(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes() if p.exists() else b''

old_supervisor=json.loads((old/'supervisor.json').read_text())
assert not command(old_supervisor['pid']), 'v41 supervisor must exit first'
for rank in (0,1):
    status=json.loads((old/f'rank{rank}.status.json').read_text())
    assert b'src_stable_onpolicy_v41/train.py' not in command(status['pid'])
out.mkdir(exist_ok=True)
if (out/'supervisor.json').exists():
    assert not command(json.loads((out/'supervisor.json').read_text())['pid']), 'v42 is already live'
    assert not (out/'latest.COMMITTED.json').exists(), 'Use resume_owned.py for an existing run'
contracts=json.loads((source/'grounding_contracts.json').read_text())
for key in ('independent_predictor_parameters','future_evidence_cannot_change_earlier_grounded_state',
            'target_labels_independent_of_predictor','no_grounding_encoder_in_rule_compilation',
            'actual_training_batch_backward_finite','raw_visual_evidence_grounding'):
    assert contracts[key]
cache=ROOT/'onpolicy_successors_v40'
assert json.loads((cache/'status.json').read_text())['state']=='complete'
readiness=json.loads((ROOT/'src_stable_onpolicy_v41/data_readiness.json').read_text())
assert readiness['ready'] and readiness['all_hashes_verified']
if not (source/'predictive_nesy').exists():
    shutil.copytree(ROOT/'src_stable_onpolicy_v41/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
commit=json.loads((old/'best.COMMITTED.json').read_text());checkpoint=old/commit['filename']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']
record=dict(source_checkpoint=str(checkpoint),source_commit=commit,created=time.time(),
    previous_final_commit=json.loads((old/'latest.COMMITTED.json').read_text()),
    data_cache=str(cache),gpus=[6,7],contracts=contracts,
    effect_on_navigation='Unproven. Evaluate unchanged full dev80 and retain independent finalfixed80.')
(out/'launcher.json').write_text(json.dumps(record,indent=2))
with (out/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(source/'supervisor.py')],cwd=source,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,time=time.time(),source_commit=commit,run=str(out))))
