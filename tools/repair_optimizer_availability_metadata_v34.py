"""Remove stale resume-availability pointers after authorized cold optimizer pruning."""
import json,os,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');reference=Path('/home/user/xjy/strips_wm_navigation_20261009')
read=lambda p:json.loads(p.read_text())
pruned=set()
for name in ['obsolete_auxiliary_optimizer_compaction_v34.json','obsolete_classifier_optimizer_compaction_v34.json']:
    audit=read(base/name);assert audit['state']=='complete'
    pruned.update(row['folder'] for row in audit['records'])
records=[]
for root in [base,reference/'runs']:
    for marker in root.rglob('COMMITTED'):
        old=read(marker)
        if old.get('full_optimizer_available_at') not in pruned:continue
        parent=marker.parent;assert parent.resolve().is_relative_to(root.resolve())
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            if proc.parent.name==str(os.getpid()):continue
            try:assert str(parent).encode() not in proc.read_bytes(),'Referenced checkpoint currently in use'
            except OSError:pass
        new=dict(old);source=new.pop('full_optimizer_available_at');new['former_full_optimizer_source_pruned']=source
        for path in [marker,parent/'metadata.json']:
            if not path.exists():continue
            value=read(path)
            if value.get('full_optimizer_available_at')!=source:continue
            value.pop('full_optimizer_available_at');value['former_full_optimizer_source_pruned']=source
            temp=path.with_name(path.name+'.availability.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
        records.append({'folder':str(parent),'old_commit':old,'new_commit':new,'checkpoint_bytes_changed':False})
result={'time':time.time(),'records':records,'checkpoint_weights_unchanged':True}
(base/'optimizer_availability_metadata_v34.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'corrected_pointers':len(records)}))
