"""Remove only verified obsolete evaluation snapshots from the reviewed plan."""
import hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
plan_path=base/'retired_evaluation_pruning_plan.json';plan=json.loads(plan_path.read_text())
audit_path=base/'retired_evaluation_pruning_applied.json'
assert not audit_path.exists(), 'Already applied; inspect audit rather than repeat'
allowed={'v3','discrete_v4','symbolic_sft_v5','symbolic_robust_v7','symbolic_vision_v8','symbolic_joint_v9',
         'symbolic_reward_v11','symbolic_task_v15','symbolic_branch_v17'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def idle(run):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:
            assert str(run).encode() not in p.read_bytes(), 'Run is active'
        except OSError:pass
    status=json.loads((run/'status.json').read_text())
    assert status['state']=='complete' or (run/'STOP').exists()
checked=[]
for item in plan['items']:
    run=Path(item['run']).resolve();folder=Path(item['folder']).resolve();path=Path(item['file']).resolve()
    assert run.is_relative_to(base) and run.relative_to(base).parts[0] in allowed
    assert folder.parent.parent==run/'evaluations' and folder.name=='checkpoint'
    assert path.parent==folder and not Path(item['file']).is_symlink()
    assert item['commit']['epoch'] not in item['retained_epochs']
    assert str(folder) not in plan['protected_evaluation_checkpoints']
    idle(run)
    assert json.loads((folder/'COMMITTED').read_text())==item['commit']
    assert sha(path)==item['commit']['checkpoint_sha256']
    for retained in item['retained_checkpoint_folders']:
        keep=Path(retained);record=json.loads((keep/'COMMITTED').read_text())
        assert (keep/record['checkpoint_file']).is_file()
    checked.append(item)
audit={'time':time.time(),'plan_sha256':sha(plan_path),'free_before':shutil.disk_usage(base).free,
       'items':checked,'completed':[],'state':'validated_before_removal'}
audit_path.write_text(json.dumps(audit,indent=2))
for item in checked:
    run=Path(item['run']);folder=Path(item['folder']);path=Path(item['file']);idle(run)
    assert json.loads((folder/'COMMITTED').read_text())==item['commit']
    (folder/'PRUNED_CHECKPOINT.json').write_text(json.dumps({'time':time.time(),'original_commit':item['commit'],
        'reason':'Obsolete intermediate snapshot; first, best, final and all trajectory evidence retained.',
        'retained_checkpoint_folders':item['retained_checkpoint_folders']},indent=2))
    (folder/'COMMITTED').unlink()
    alias=folder/'checkpoint.pt'
    if alias.is_symlink():
        assert alias.resolve()==path;alias.unlink()
    path.unlink()
    audit['completed'].append(str(folder));audit_path.write_text(json.dumps(audit,indent=2))
audit.update(state='complete',finished=time.time(),free_after=shutil.disk_usage(base).free)
audit_path.write_text(json.dumps(audit,indent=2))
print(json.dumps({'removed_snapshots':len(checked),'free_before_gib':audit['free_before']/2**30,
    'free_after_gib':audit['free_after']/2**30,'audit':str(audit_path)}))
