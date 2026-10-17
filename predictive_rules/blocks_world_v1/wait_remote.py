"""Wait on process-exit notification, without repeatedly polling training."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from predictive_remote import connect

remote = r'''
import json,os,pathlib,select,time
p=pathlib.Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/blocks_world_v1/run_20261016')
r=json.loads((p/'launcher.json').read_text());pid=r['pid']
try:
 fd=os.pidfd_open(pid)
except ProcessLookupError:
 fd=None
if fd is not None:
 command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert str(p) in command or not command
 started=time.time()
 while time.time()-started<3600:
  if select.select([fd],[],[],50)[0]:break
  print(json.dumps({'event':'waiting_on_process_exit','elapsed_seconds':round(time.time()-started)}),flush=True)
 os.close(fd)
print('FINAL_STATUS', (p/'status.json').read_text() if (p/'status.json').exists() else 'missing',flush=True)
print('LOG_TAIL', '\n'.join((p/'driver.log').read_text(errors='replace').splitlines()[-12:]),flush=True)
if (p/'comparison.json').exists():print('COMPARISON', (p/'comparison.json').read_text(),flush=True)
'''

client=connect()
try:
 _,stdout,stderr=client.exec_command("python3 -u - <<'WAIT_BLOCKS'\n"+remote+"\nWAIT_BLOCKS",timeout=120)
 for line in stdout:print(line,end='',flush=True)
 err=stderr.read().decode()
 if err:print(err,flush=True)
 raise SystemExit(stdout.channel.recv_exit_status())
finally:client.close()
