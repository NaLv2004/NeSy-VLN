"""Stop only the identified GRPO run after the user's validation condition."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

B = Path('/home/user/xjy/nesy_vla_20261010/group_grpo_v98')
read = lambda p: json.loads(p.read_text())

def proc(pid):
    try:
        p = Path('/proc') / str(pid)
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=stat[0], ppid=int(stat[1]),
                    pgid=int(stat[2]), sid=int(stat[3]), start=stat[19],
                    command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None

evaluations = {p.parent.name: read(p) for p in sorted((B / 'evaluations').glob('*/summary.json'))}
latest_validation = evaluations[sorted(evaluations)[-1]]
source = read(B / 'SOURCE.json')['selected_development']
assert source['successes'] == 43
assert max(v['successes'] for v in evaluations.values()) <= source['successes'], 'An improved validation requires reassessment'
jobs = read(B / 'launcher.json')['jobs']
groups = []
for j in jobs:
    folder = B / f"rank{j['rank']}"
    supervisor = read(folder / 'supervisor.json')
    pgid = j['supervisor_pid']
    s, t = proc(pgid), proc(supervisor['pid'])
    if s and s['state'] != 'Z':
        assert str(B / 'supervise.py') in s['command'] and s['pgid'] == pgid and s['sid'] == pgid
    if t and t['state'] != 'Z':
        assert str(B / 'src/train.py') in t['command'] and t['pgid'] == pgid and t['sid'] == pgid
    for pid in read(folder / 'processes.json')['simulator_pids']:
        p = proc(pid)
        if p and p['state'] != 'Z':
            assert p['pgid'] == pgid and p['sid'] == pgid, ('Unexpected simulator group', p)
    groups.append(pgid)

members = [p for d in Path('/proc').iterdir() if d.name.isdigit()
           for p in [proc(int(d.name))] if p and p['pgid'] in groups and p['state'] != 'Z']
assert all(p['sid'] == p['pgid'] for p in members)
report = dict(time=time.time(), user_requested=True, do_not_restart=True,
              reason='Five scheduled validations failed to improve over the 43/80 source; user requested termination if no improvement.',
              latest_validation=latest_validation, validation_successes={k:v['successes'] for k,v in evaluations.items()},
              checkpoint_before=read(B / 'latest/COMMITTED'), process_groups=groups,
              process_count=len(members), processes=members)
(B / 'USER_STOP.json').write_text(json.dumps(report, indent=2))
for pgid in groups:
    try: os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError: pass

def remaining():
    return [p for old in members for p in [proc(old['pid'])]
            if p and p['start'] == old['start'] and p['state'] != 'Z']

deadline = time.monotonic() + 10
while remaining() and time.monotonic() < deadline:
    time.sleep(0.25)
for p in remaining():
    os.kill(p['pid'], signal.SIGKILL)
deadline = time.monotonic() + 5
while remaining() and time.monotonic() < deadline:
    time.sleep(0.25)
report['remaining'] = remaining()
report['checkpoint_preserved'] = read(B / 'latest/COMMITTED')
commit = report['checkpoint_preserved']
report['checkpoint_sha256_verified'] = hashlib.sha256((B / 'latest' / commit['checkpoint_file']).read_bytes()).hexdigest() == commit['checkpoint_sha256']
source_commit = read(B / 'source_snapshot/COMMITTED')
report['source_sha256_verified'] = hashlib.sha256((B / 'source_snapshot' / source_commit['checkpoint_file']).read_bytes()).hexdigest() == source_commit['checkpoint_sha256']
report['gpus_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
report['gpu_processes_after'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
report['finished_at'] = time.time()
for j in jobs:
    folder = B / f"rank{j['rank']}"
    for filename in ['supervisor.json', 'status.json']:
        p = folder / filename
        s = read(p)
        s.update(state='stopped_by_user', time=report['finished_at'], stop_record=str(B / 'USER_STOP.json'), do_not_restart=True)
        tmp = p.with_suffix('.stop.tmp')
        tmp.write_text(json.dumps(s, indent=2)); os.replace(tmp, p)
(B / 'USER_STOP.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'processes'}, indent=2))
assert not report['remaining']
assert report['checkpoint_sha256_verified'] and report['source_sha256_verified']
