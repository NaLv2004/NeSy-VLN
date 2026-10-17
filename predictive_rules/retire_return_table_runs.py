"""Retire only superseded owned runs, preserving verified training snapshots."""
import hashlib,json,os,signal,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
targets=[('run_anonymous_v5','train_anonymous.py'),('run_anonymous_v6','train_anonymous.py'),
    ('counterfactual_anonymous_v5/shard_0','collect_anonymous_outcomes.py'),
    ('counterfactual_anonymous_v5/shard_1','collect_anonymous_outcomes.py')]
records=[]
for relative,expected in targets:
    folder=root/relative;launch=folder/'launcher.json'
    if not launch.exists():continue
    pid=json.loads(launch.read_text())['pid'];proc=Path(f'/proc/{pid}')
    if not proc.exists():records.append(dict(run=relative,pid=pid,already_terminal=True));continue
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=(proc/'cwd').resolve()
    if expected not in command or str(root) not in command or not cwd.is_relative_to(root):raise RuntimeError('Process identity differs')
    if os.getpgid(pid)!=pid:raise RuntimeError('Refuse to signal a shared process group')
    saved=None
    if relative.startswith('run_'):
        for _ in range(3):
            before=json.loads((folder/'latest.COMMITTED.json').read_text());data=(folder/'latest.pt').read_bytes()
            after=json.loads((folder/'latest.COMMITTED.json').read_text())
            if before==after and hashlib.sha256(data).hexdigest()==before['sha256']:break
        else:raise RuntimeError('No consistent saved checkpoint; preserve live process')
        (folder/'retired_checkpoint.pt').write_bytes(data)
        saved={**before,'path':'retired_checkpoint.pt'};(folder/'retired_checkpoint.COMMITTED.json').write_text(json.dumps(saved,indent=2))
    os.killpg(pid,signal.SIGTERM)
    deadline=time.time()+15
    while proc.exists() and time.time()<deadline:
        if (proc/'stat').read_text().split()[2]=='Z':break
        time.sleep(.1)
    alive=proc.exists() and (proc/'stat').read_text().split()[2]!='Z'
    if alive:raise RuntimeError(f'Owned process {pid} did not exit after SIGTERM; inspect before escalation')
    record=dict(run=relative,pid=pid,time=time.time(),state='retired_superseded',preserved_checkpoint=saved,
        reason='User explicitly rejects reward/value-table planning; replace with network-generated anonymous subgoal and symbolic reachability')
    (folder/'retired.json').write_text(json.dumps(record,indent=2));records.append(record)
print(json.dumps(records,indent=2))
