import json,os,time,collections
from pathlib import Path
base=Path('/home/user/xjy/strips_wm_navigation_20261009/runs').resolve();report=[]
def live(run):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:
            if str(run).encode() in p.read_bytes():return True
        except OSError:pass
    return False
for run in sorted(base.iterdir()):
    if not run.is_dir() or live(run):continue
    commits=[]
    for marker in run.rglob('COMMITTED'):
        try:record=json.loads(marker.read_text())
        except Exception:continue
        name=record.get('checkpoint_file','checkpoint.pt');p=marker.parent/name
        if p.is_file():commits.append({'folder':str(marker.parent),'name':str(marker.parent.relative_to(run)),
            'record':record,'bytes':p.stat().st_size,'links':p.stat().st_nlink})
    if len(commits)>2:report.append({'run':str(run),'checkpoints':commits})
out=Path('/home/user/xjy/nesy_vla_20261010/retired_strips_intermediates_v34.json')
out.write_text(json.dumps({'time':time.time(),'runs':report},indent=2))
print(json.dumps([{'run':r['run'],'checkpoints':[{'name':c['name'],'mib':round(c['bytes']/2**20,1),
    'epoch':c['record'].get('epoch'),'links':c['links']} for c in r['checkpoints']]} for r in report],indent=2))
