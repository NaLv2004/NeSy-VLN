from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'scripts/compact_intermediate_optimizers_v46.py').read_text()
source=source.replace("allowed={'symbolic_task_v15','symbolic_branch_v17','symbolic_teacher_v25'}", "allowed={'symbolic_execution_v38'}")
source=source.replace('obsolete_intermediate_optimizer_compaction_v46.json','redundant_best_optimizer_compaction_v61.json')
source=source.replace("folder=marker.parent;run=folder.parent", """folder=marker.parent;run=folder.parent
        if folder.name!='best_development':continue
        latest=run/'latest/COMMITTED'
        if not latest.exists():continue
        latest_commit=read_json(latest);best_commit=read_json(marker)
        if latest_commit.get('epoch')!=best_commit.get('epoch'):continue
        if not latest_commit.get('optimizer_included'):continue
        assert sha256(latest.parent/latest_commit['checkpoint_file'])==latest_commit['checkpoint_sha256']
        status=read_json(run/'status.json')
        if status.get('state')!='complete':continue""")
source=source.replace("payload['optimizer']=None", """payload['optimizer']=None
        full=load_committed(run/'latest')
        assert same(payload['model'],full['model']), 'Best/latest model mismatch'
        del full""")
source=source.replace("new=dict(metadata,checkpoint_file=", "metadata['full_optimizer_available_at']=str(run/'latest')\n        new=dict(metadata,checkpoint_file=")
(root/'scripts/compact_redundant_best_optimizer_v61.py').write_text(source)
