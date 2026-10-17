from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'scripts/compact_intermediate_optimizers_v46.py').read_text()
source=source.replace("allowed={'symbolic_task_v15','symbolic_branch_v17','symbolic_teacher_v25'}","allowed={'symbolic_execution_v38'}")
source=source.replace("'obsolete_intermediate_optimizer_compaction_v46.json'","'retired_execution_optimizer_compaction_v61.json'")
needle="folder=marker.parent;run=folder.parent"
assert needle in source
source=source.replace(needle,needle+"\n        if folder.name not in ['step_000020','step_000030']:continue\n        if not (run/'status.json').exists() or read_json(run/'status.json').get('state')!='complete':continue\n        if (run/'best_development/COMMITTED').exists() and read_json(run/'best_development/COMMITTED').get('epoch')==read_json(marker).get('epoch'):continue")
source=source.replace("if shutil.disk_usage(base).free>=8*2**30:break","if shutil.disk_usage(base).free>=4.6*2**30:break")
source=source.replace("'symbolic_recovery_v28','symbolic_execution_v38','token_revision_rl_v45'","'symbolic_recovery_v28','token_revision_rl_v45','factorized_automaton_v51','goal_conditioned_automaton_v55','probabilistic_automaton_v56'")
source=source.replace("'all_model_weights_and_training_validation_evidence_retained':True","'all_model_weights_and_training_validation_evidence_retained':True,'protected_v38_checkpoints':'initial,first,best,final,latest. Only unshared intermediate20/30 optimizer slots removed, weights and all other payload fields retained exactly.'")
(root/'scripts/compact_retired_execution_optimizers_v61.py').write_text(source,encoding='utf-8')
print('Prepared narrow retired v38 intermediate optimizer compaction')
