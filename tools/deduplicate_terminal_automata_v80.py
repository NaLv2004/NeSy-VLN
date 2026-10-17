"""Coalesce byte-identical cold checkpoint files; preserve all names and contents."""
import hashlib,json,os,shutil,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010').resolve()
families=['goal_conditioned_automaton_v55','probabilistic_automaton_v56','full_automaton_rl_v62',
          'higher_order_capacity_v64','perceptual_fsm_v74','observation_inverse_v76','sequence_perception_v78']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def active(root):
 for f in Path('/proc').glob('[0-9]*/cmdline'):
  if f.parent.name==str(os.getpid()):continue
  try:
   if str(root).encode() in f.read_bytes():return True
  except OSError:pass
 return False
before=shutil.disk_usage(b).free;seen={};records=[];checked=[];skipped=[]
for family in families:
 root=b/family
 if not root.exists():continue
 if active(root):skipped.append(str(root));continue
 for marker in sorted(root.rglob('COMMITTED')):
  meta=json.loads(marker.read_text());path=marker.parent/meta.get('checkpoint_file','checkpoint.pt')
  if path.is_file() and not path.is_symlink():assert digest(path)==meta['checkpoint_sha256']
 for path in sorted(root.rglob('checkpoint*.pt')):
  if path.is_symlink() or not path.is_file():continue
  assert path.resolve().is_relative_to(root)
  stat=path.stat();key=(stat.st_size,digest(path));prior=seen.get(key)
  if prior and prior.stat().st_ino!=stat.st_ino:
   assert not active(root) and not active(next((b/f for f in families if prior.is_relative_to(b/f))))
   assert digest(prior)==key[1] and path.stat().st_mtime_ns==stat.st_mtime_ns
   temp=path.with_name(path.name+'.dedup_link');assert not temp.exists()
   os.link(prior,temp);os.replace(temp,path);assert digest(path)==key[1]
   records.append({'path':str(path),'canonical':str(prior),'sha256':key[1],'bytes':stat.st_size})
  else:seen[key]=path
  checked.append(str(path))
r={'time':time.time(),'files_checked':len(checked),'records':records,'skipped_live':skipped,
   'free_before_gib':before/2**30,'free_after_gib':shutil.disk_usage(b).free/2**30,
   'preserved':'Every checkpoint path and its byte contents; model, optimizer and RNG unchanged.'}
(b/'terminal_checkpoint_deduplication_v80.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
