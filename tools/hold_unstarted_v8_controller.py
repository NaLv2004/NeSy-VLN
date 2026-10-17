"""Hold an unstarted allocation only while its new objective is corrected."""
import json,os,signal,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_vision_v8')
record=json.loads((root/'controller.json').read_text());pid=record['pid'];proc=Path(f'/proc/{pid}/cmdline')
assert proc.exists() and str(root/'start_when_ready.py') in proc.read_bytes().replace(b'\0',b' ').decode()
os.kill(pid,signal.SIGSTOP)
try:
    allocation=json.loads((root/'allocation.json').read_text())
    assert allocation['jobs']=={},'A new training has already started; do not modify its frozen source'
    assert not list((root/'runs').glob('*/supervisor.json'))
    os.kill(pid,signal.SIGTERM)
finally:os.kill(pid,signal.SIGCONT)
deadline=time.time()+20
while proc.exists() and time.time()<deadline:time.sleep(.5)
assert not proc.exists(),'Allocation controller still present'
record.update(held_at=time.time(),state='held_before_any_new_training',reason='Make inverse action prediction depend on discrete predicted states, not soft confidence differences.')
(root/'controller_held.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
