"""Plan removal of obsolete intermediate weights; keep first, best and final models."""
import hashlib,json,os,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve();rows=[];protected=[]
names=['runs/full_v1','v2/runs/full_v2','v3/runs/full_v3','discrete_v4/runs/discrete_gru','discrete_v4/runs/discrete_vlm',
    'symbolic_sft_v5/runs/sft_plain','symbolic_sft_v5/runs/sft_spatial',
    'symbolic_robust_v7/runs/robust_reference','symbolic_robust_v7/runs/robust_multieffect',
    'symbolic_vision_v8/runs/frozen_convnext','symbolic_vision_v8/runs/frozen_qwen',
    'symbolic_joint_v9/runs/joint','symbolic_reward_v11/runs/original','symbolic_reward_v11/runs/bounded_repeat_stop40',
    'symbolic_task_v15/runs/paired_tasks','symbolic_branch_v17/runs/branch_current']
def live(run):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:
            if str(run).encode() in p.read_bytes():return True
        except OSError:pass
    return False
for name in names:
    run=base/name
    if not (run/'status.json').exists() or live(run):continue
    status=json.loads((run/'status.json').read_text())
    if status.get('state')!='complete' and not (run/'STOP').exists():continue
    retain={status['update']};retained=[]
    for key in ['latest','best','best_development']:
        marker=run/key/'COMMITTED'
        if marker.exists():
            record=json.loads(marker.read_text());retain.add(record['epoch']);retained.append(str(marker.parent))
    markers=sorted((run/'evaluations').glob('step_*/checkpoint/COMMITTED'))
    if markers:retain.add(int(markers[0].parent.parent.name.split('_')[-1]))
    if name=='symbolic_robust_v7/runs/robust_multieffect':retain.add(80)
    for marker in markers:
        old=json.loads(marker.read_text());folder=marker.parent
        if old['epoch'] in retain:protected.append(str(folder));continue
        path=folder/old['checkpoint_file']
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base)
        rows.append({'run':str(run),'folder':str(folder),'commit':old,'file':str(path),
            'bytes':path.stat().st_size,'reclaimable_bytes':path.stat().st_size if path.stat().st_nlink==1 else 0,
            'retained_checkpoint_folders':retained,'retained_epochs':sorted(retain)})
plan={'time':time.time(),'root':str(base),'items':rows,'protected_evaluation_checkpoints':protected,
    'purpose':'User-authorized removal of obsolete intermediate training checkpoints; all traces, original data, first/best/final and globalbest22/80 weights retained.'}
(base/'retired_evaluation_pruning_plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps({'candidates':len(rows),'reclaimable_gib':sum(r['reclaimable_bytes'] for r in rows)/2**30,
    'protected':protected,'candidate_folders':[r['folder'] for r in rows]},indent=2))
