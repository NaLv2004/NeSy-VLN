import json,time,shutil,subprocess
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'run_boundary_balanced_v43'
def read(name):return json.loads((out/name).read_text())
supervisor=read('supervisor.json');statuses=[read(f'rank{rank}.status.json') for rank in (0,1)]
pids=[supervisor['pid'],supervisor['worker_pid']]+[s['pid'] for s in statuses]
commands=[]
for pid in pids:
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'src_boundary_balanced_v43/' in command
    commands.append(dict(pid=pid,command=command))
manifest=read('manifest.json');launcher=read('launcher.json')
assert manifest['resume_commit']==launcher['source_commit']
assert all(s['update']>launcher['source_commit']['update'] for s in statuses)
assert all(s['branch_initial_fraction']==.25 and s['branch_early_fraction']==.25 and s['branch_late_fraction']==.5 for s in statuses)
record=dict(time=time.time(),processes=commands,statuses=statuses,
    manifest_resume=manifest['resume_commit'],source_architecture=manifest['resume_architecture'],
    free_gib=shutil.disk_usage(ROOT).free/2**30,
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
(out/'startup_verified.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
