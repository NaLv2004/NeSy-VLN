"""Losslessly archive retired evaluation snapshots; keep live/best resumes online."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010').resolve();audit=b/'cold_evaluation_weight_archives_v61.json'
assert not audit.exists(), 'Inspect the existing archive audit before continuing'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def live(run):
 for f in Path('/proc').glob('[0-9]*/cmdline'):
  if f.parent.name==str(os.getpid()):continue
  try:
   if str(run).encode() in f.read_bytes():return True
  except OSError:pass
 return False
records=[]
for family in ['symbolic_execution_v38','symbolic_recovery_v28']:
 for run in sorted((b/family/'runs').glob('*')):
  status=run/'status.json'
  if not status.exists() or json.loads(status.read_text()).get('state')!='complete' or live(run):continue
  assert (run/'latest/COMMITTED').exists() and (run/'best_development/COMMITTED').exists()
  for marker in sorted((run/'evaluations').glob('step_*/checkpoint/COMMITTED')):
   commit=json.loads(marker.read_text());path=marker.parent/commit['checkpoint_file']
   if not path.exists() or path.is_symlink() or path.stat().st_nlink!=1:continue
   assert path.resolve().is_relative_to(b/family) and sha(path)==commit['checkpoint_sha256']
   archive=path.with_suffix('.pt.zst');temporary=archive.with_suffix('.partial')
   assert not archive.exists() and not temporary.exists()
   subprocess.run(['zstd','-q','-3','-T2',str(path),'-o',str(temporary)],check=True)
   with subprocess.Popen(['zstd','-q','-d','-c',str(temporary)],stdout=subprocess.PIPE) as process:
    h=hashlib.sha256();size=0
    for block in iter(lambda:process.stdout.read(8<<20),b''):h.update(block);size+=len(block)
    assert process.wait()==0
   assert h.hexdigest()==commit['checkpoint_sha256'] and size==path.stat().st_size
   assert json.loads(marker.read_text())==commit and not live(run)
   temporary.replace(archive)
   row={'source':str(path),'archive':str(archive),'original_sha256':h.hexdigest(),'original_bytes':size,
    'compressed_bytes':archive.stat().st_size,'restore_argv':['zstd','-d','-k',str(archive),'-o',str(path)],'time':time.time()}
   (marker.parent/'ARCHIVED_CHECKPOINT.json').write_text(json.dumps(row,indent=2))
   records.append(row);audit.write_text(json.dumps({'state':'running','pid':os.getpid(),'records':records},indent=2))
   path.unlink()
   if shutil.disk_usage(b).free>4.5*2**30:break
  if shutil.disk_usage(b).free>4.5*2**30:break
 if shutil.disk_usage(b).free>4.5*2**30:break
result={'state':'complete','pid':os.getpid(),'records':records,'time':time.time(),
 'released_gib':sum(r['original_bytes']-r['compressed_bytes'] for r in records)/2**30,'free_gib':shutil.disk_usage(b).free/2**30}
audit.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
