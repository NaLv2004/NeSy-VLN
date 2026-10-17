import json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010')
root=Path('/home/user/xjy/strips_wm_navigation_20261009')
cfg=json.loads((root/'configs/experiment.json').read_text())
rows=json.loads((Path(cfg['source_data'])/'train.json').read_text())
lock=json.loads((root/'research/split_lock.json').read_text())
allowed=set(lock['train_scenes'])-set(lock['development_scenes'])
sample=next(r for r in rows if r['scene'] in allowed)
same=[r for r in rows if r['scene']==sample['scene'] and r['category']!=sample['category']][:2]
recovery=json.loads((base/'storage_recovery_verification.json').read_text())
if recovery['all_actual_updates_recovered']:
    hold=json.loads((base/'storage_hold_20261011.json').read_text())
    hold.update(state='recovered',verified_recovery_time=recovery['time'])
    (base/'storage_hold_20261011.json').write_text(json.dumps(hold,indent=2))
print(json.dumps({'time':time.time(),'config':cfg,'sample':sample,'other_goal':same,
    'recovery':recovery,'archive_restore':json.loads((base/'evidence_archives/restore_verification.json').read_text())},indent=2))
