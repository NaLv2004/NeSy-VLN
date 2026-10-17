import os,signal
from pathlib import Path
stopped=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'audit_training_states.py' in args and b'../run_planning_observed_v29/checkpoint_002000.pt' in args:
        os.kill(int(p.name),signal.SIGTERM);stopped.append(int(p.name))
print({'diagnostic_processes_stopped':stopped,'training_untouched':True})
