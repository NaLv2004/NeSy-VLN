"""Retire only the owned v4 trainer after the diagnosed world-model failure."""
import hashlib,json,os,shutil,signal,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'run_graph_subgoal_v4';src=root/'src_graph_subgoal_v4'
record=json.loads((out/'launcher.json').read_text());pid=record['pid'];proc=Path('/proc')/str(pid)
if proc.exists():
    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=(proc/'cwd').resolve();pgid=os.getpgid(pid)
    if str(src/'train_graph_subgoal.py') not in cmd or cwd!=src or pgid!=pid:raise RuntimeError('Owned process identity mismatch')
    os.killpg(pgid,signal.SIGTERM)
    for _ in range(30):
        if not proc.exists():break
        time.sleep(.25)
    if proc.exists():raise RuntimeError('Termination still pending; no restart or forced kill performed')
commit=json.loads((out/'latest.COMMITTED.json').read_text());path=out/commit['path']
if hashlib.sha256(path.read_bytes()).hexdigest()!=commit['sha256']:raise RuntimeError('Committed checkpoint mismatch')
dest=out/'retired_checkpoint.pt'
if not dest.exists():shutil.copy2(path,dest)
if hashlib.sha256(dest.read_bytes()).hexdigest()!=commit['sha256']:raise RuntimeError('Retired archive mismatch')
archived=dict(commit,path=dest.name);(out/'retired_checkpoint.COMMITTED.json').write_text(json.dumps(archived,indent=2))
retired=dict(time=time.time(),pid=pid,checkpoint=archived,
    reason='Low navigation success plus 35/128 one-step state agreement; teacher-goal substitution did not improve first-action accuracy',
    evaluation_processes='left to finish existing checkpoints',no_unrelated_process_terminated=True)
(out/'retired.json').write_text(json.dumps(retired,indent=2));print(json.dumps(retired))
