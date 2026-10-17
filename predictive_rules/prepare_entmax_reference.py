"""Install a small pinned reference library into an isolated experiment path."""
import hashlib,importlib.metadata,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
target=ROOT/'vendor_entmax_1_3'
if not (target/'entmax/__init__.py').is_file():
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',
        '--index-url','https://mirrors.aliyun.com/pypi/simple','--target',str(target),'entmax==1.3'],check=True)
sys.path.insert(0,str(target))
import entmax
assert Path(entmax.__file__).resolve().is_relative_to(target)
files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*.py')}
record=dict(library='entmax',version='1.3',upstream='https://github.com/deep-spin/entmax',
    paper='https://www.jmlr.org/papers/v21/19-021.html',path=str(target),files=files,time=time.time(),
    isolated=True,live_training_environment_modified=False)
(target/'provenance.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
