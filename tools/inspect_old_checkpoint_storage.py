import json,subprocess
from pathlib import Path
root=Path('/home/user/xjy/strips_wm_navigation_20261009/runs')
print(subprocess.check_output(['du','-x','-B1','--max-depth=2',str(root)],text=True))
for d in sorted(root.iterdir()):
    if not d.is_dir():continue
    files=[p for p in d.rglob('*') if p.is_file() and p.suffix in ['.pt','.pth','.safetensors']]
    print(json.dumps({'run':d.name,'files':[(str(p.relative_to(d)),p.stat().st_size) for p in files][:100],
      'markers':[(str(p.relative_to(d)),p.read_text()[:300]) for p in d.glob('*status*.json')]}))
