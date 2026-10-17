"""Retain all historical Qwen adapters; prune only obsolete optimizer states.

Three newest complete checkpoints per retired run remain fully resumable.
Every older adapter, config, RNG file and evaluation stays in place.
"""
import hashlib,json,os,shutil,time
from pathlib import Path
import torch
base=Path('/home/user/xjy/objectnav_qwen35_rl_20260921/runs').resolve()
audit=Path('/home/user/xjy/nesy_vla_20261010/cold_qwen_optimizer_cleanup_v81.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def live(run):
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  if p.parent.name==str(os.getpid()):continue
  try:
   if str(run).encode() in p.read_bytes():return True
  except OSError:pass
 return False
records=[];before=shutil.disk_usage(base).free
for run in sorted(base.iterdir()):
 if not run.is_dir() or run.is_symlink() or live(run):continue
 checkpoints=sorted(p for p in run.glob('step_[0-9][0-9][0-9][0-9][0-9][0-9]') if (p/'COMMITTED').exists())
 resumable=[p for p in checkpoints if (p/'trainer.pt').is_file() and (p/'adapter/adapter_model.safetensors').is_file()]
 if len(resumable)<=3:continue
 keep=set(resumable[-3:])
 # Explicit latest pointer is retained even if a run uses an unusual ordering.
 latest=run/'latest'
 if latest.is_symlink():keep.add(latest.resolve())
 elif latest.is_file():
  try:
   name=latest.read_text().strip()
   if (run/name).is_dir():keep.add((run/name).resolve())
  except (UnicodeError,OSError):pass
 for folder in resumable:
  if folder in keep or folder.resolve() in keep:continue
  path=folder/'trainer.pt';adapter=folder/'adapter/adapter_model.safetensors'
  assert path.resolve().is_relative_to(base) and not path.is_symlink() and path.stat().st_nlink==1
  state=torch.load(path,map_location='cpu',weights_only=False)
  # Never remove model weights hidden in an unconventional checkpoint format.
  if set(state)!={'step','optimizer'} or not isinstance(state['optimizer'],dict):continue
  assert int((folder/'COMMITTED').read_text())==state['step']
  del state
  assert all((p/'trainer.pt').is_file() and (p/'COMMITTED').exists() for p in keep)
  assert not live(run)
  entry={'time':time.time(),'path':str(path),'bytes':path.stat().st_size,'optimizer_sha256':sha(path),
         'adapter':str(adapter),'adapter_sha256':sha(adapter),
         'full_resume_retained':[str(p) for p in sorted(keep)],'inference_weights_preserved':True}
  records.append(entry);audit.write_text(json.dumps({'state':'cleaning','records':records},indent=2))
  (folder/'INFERENCE_ONLY.json').write_text(json.dumps(entry,indent=2))
  path.unlink();assert adapter.is_file() and sha(adapter)==entry['adapter_sha256']
 print(json.dumps({'run':run.name,'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
r={'state':'complete','time':time.time(),'records':records,'removed_gib':sum(r['bytes'] for r in records)/2**30,
   'free_before_gib':before/2**30,'free_after_gib':shutil.disk_usage(base).free/2**30,
   'preserved':'All model adapters, original data, evaluation evidence, configs, RNG files; latest three full optimizer checkpoints per retired run.'}
audit.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
