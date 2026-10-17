"""Remove recreatable package download caches without touching installed environments."""
import json,os,shutil,time
from pathlib import Path
b=Path('/home/user/xjy').resolve();out=b/'nesy_vla_20261010/install_cache_cleanup_v81.json'
targets=[b/'objectnav_qwen35_rl_20260921/cache/pip',b/'vamos_4x4090_20260906/cache/uv']
commands=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 if p.parent.name==str(os.getpid()):continue
 try:commands.append(p.read_bytes().replace(b'\0',b' ').decode(errors='replace'))
 except OSError:pass
records=[];before=shutil.disk_usage(b).free
for target in targets:
 assert target.resolve().is_relative_to(b) and target.name in ('pip','uv') and target.parent.name=='cache'
 if not target.exists():continue
 assert not target.is_symlink()
 assert not any(str(target) in cmd for cmd in commands)
 # uv uses hard links by default; deleting one link preserves installed files.
 # A nonstandard symlink installation into the cache must be preserved instead.
 project=target.parent.parent
 symlinks=[]
 for folder in [project/'env',project/'envs']:
  if not folder.exists():continue
  for root,dirs,files in os.walk(folder,followlinks=False):
   for name in dirs+files:
    p=Path(root)/name
    if p.is_symlink() and p.resolve().is_relative_to(target):symlinks.append(str(p))
 assert not symlinks,symlinks[:10]
 files=[p for p in target.rglob('*') if p.is_file() and not p.is_symlink()]
 record={'path':str(target),'files':len(files),'logical_bytes':sum(p.stat().st_size for p in files),
         'expected_released_bytes':sum(p.stat().st_size for p in files if p.stat().st_nlink==1),'time':time.time()}
 records.append(record);out.write_text(json.dumps({'state':'cleaning','records':records},indent=2))
 shutil.rmtree(target)
 record['removed']=not target.exists()
result={'state':'complete','time':time.time(),'records':records,'free_before_gib':before/2**30,
        'free_after_gib':shutil.disk_usage(b).free/2**30,'original_data_models_and_installed_packages_preserved':True}
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
