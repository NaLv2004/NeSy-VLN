"""Preserve the initial preflight evidence while promoting the completed PPO source."""
import json,os
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/symbolic_ppo8_v82').resolve()
assert not (b/'launcher.json').exists()
for f in Path('/proc').glob('[0-9]*/cmdline'):
 if f.parent.name==str(os.getpid()):continue
 try:assert str(b/'src/train.py').encode() not in f.read_bytes(),'A preflight is still active'
 except OSError:pass
for name in ['SOURCE.json','PREPARED.json','preflight']:
 p=b/name;q=b/(name+'.supervised37_preflight')
 if p.exists():
  assert not q.exists() and p.resolve().is_relative_to(b) and q.parent==b
  p.rename(q)
print('Initial integration evidence retained; new source will be frozen before launch.')
