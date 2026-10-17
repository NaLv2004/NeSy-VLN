import json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/strips_wm_navigation_20261009')
result={'time':time.time(),'filesystems':subprocess.check_output(['df','-h'],text=True),
    'folders':subprocess.check_output(['du','-x','-B1','--max-depth=2',str(root/'runs')],text=True).splitlines()}
result['folders']=sorted(result['folders'],key=lambda x:int(x.split()[0]),reverse=True)[:30]
result['runs']=[]
for run in (root/'runs').iterdir():
    if not run.is_dir():continue
    commits=list(run.rglob('COMMITTED'))
    if not commits:continue
    result['runs'].append({'run':str(run),'commits':len(commits),'examples':[str(c.relative_to(run)) for c in commits[:5]]})
print(json.dumps(result,indent=2))
