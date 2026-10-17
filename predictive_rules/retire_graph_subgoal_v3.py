"""Preserve committed weights, then retire only this exact owned trainer."""
import json,os,signal,hashlib,shutil,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'run_graph_subgoal_v3'
launch=json.loads((out/'launcher.json').read_text());pid=launch['pid'];proc=Path('/proc')/str(pid)
if proc.exists():
    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(out) not in cmd or 'train_graph_subgoal.py' not in cmd or os.getpgid(pid)!=pid:raise RuntimeError('Ownership mismatch')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
verified=None
for attempt in range(5):
    commit=json.loads((out/'latest.COMMITTED.json').read_text());tmp=out/'retired_checkpoint.tmp'
    shutil.copy2(out/'latest.pt',tmp)
    if digest(tmp)==commit['sha256']:verified=commit;break
    time.sleep(.2)
if verified is None:raise RuntimeError('No coherent checkpoint copy; trainer left running')
os.replace(tmp,out/'retired_checkpoint.pt');verified['path']='retired_checkpoint.pt'
(out/'retired_checkpoint.COMMITTED.json').write_text(json.dumps(verified,indent=2))
if proc.exists():os.killpg(pid,signal.SIGTERM)
record=dict(time=time.time(),pid=pid,reason='Confirmed relaxed-versus-discrete planner mismatch on 128 training samples',
    checkpoint=verified,evaluation_processes='left to finish their already saved checkpoints',no_unrelated_process_terminated=True)
(out/'retired.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
