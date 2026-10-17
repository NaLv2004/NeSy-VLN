"""Refresh measured successors under the best v57 symbolic policy."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
for name in ('model.py','definition_rule.py','planner.py','program_archive.py','evaluate.py','data.py','objectives.py','temporal_objective.py','goal_objective.py'):
    (HERE/name).write_text((BASE/'goal_equivalence_v57'/name).read_text(encoding='utf-8'),encoding='utf-8')
(HERE/'onpolicy_worker.py').write_text((BASE/'hybrid_replay_v54/onpolicy_worker.py').read_text(encoding='utf-8'),encoding='utf-8')
s=(BASE/'hybrid_replay_v54/collect_onpolicy.py').read_text(encoding='utf-8')
s=s.replace('onpolicy_successors_v54','onpolicy_successors_v60').replace('541017','601017')
s=s.replace("policy='v52 best2000 stable exact probability derivation with task-conditioned symbols'","policy='v57 best1000: all executed actions by unchanged exact categorical table derivation'")
s=s.replace("feature_dtype='float16'","feature_dtype='float16',collection_round=3,root_without_proof_is_probed=True")
s=s.replace("np.savez_compressed(temp,features=np.stack(current[i]),successor_features=np.stack(future[i]),","np.savez_compressed(temp,features=np.stack(current[i]).astype(np.float16),successor_features=np.stack(future[i]).astype(np.float16),")
assert "requests[i]=(action,pool.submit(sims[i].call,'probe_only'))" in s
(HERE/'collect_onpolicy.py').write_text(s,encoding='utf-8')
s=(BASE/'hybrid_replay_v54/coordinate_collection.py').read_text(encoding='utf-8')
s=s.replace('onpolicy_successors_v54','onpolicy_successors_v60').replace('src_hybrid_replay_v54','src_onpolicy_refresh_v60')
(HERE/'coordinate_collection.py').write_text(s,encoding='utf-8')
print('Prepared120 training tasks;2 GPUs with4 simulators each; unchanged best-v57 inference.')
