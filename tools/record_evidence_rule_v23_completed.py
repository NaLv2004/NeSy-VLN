import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules';e=root/'evidence'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
p=read(root/'goal_progress.json');final=read(e/'evidence_rule_v23_completed.json')
assert not final['process_alive'] and final['committed_hash_verified'] and final['status']['state']=='training_budget_complete'
run=p['active_runs']['evidence_rule_v23'];run.update(last_observed_state='training_budget_complete',last_observed_update=1000,
    process_alive=False,checkpoint=final['checkpoint'],development_results=final['validation'],recent=final['recent'],
    decision='Not promoted: development5/80 then2/80. Better fitted dynamics alone did not improve navigation.')
p['transition_identifiability_v23']={k:v for k,v in read(e/'transition_identifiability_v23.json').items() if k not in ['initial_observed_branches','conflicts']}
p['state_resolution_v24']=read(e/'state_resolution_v24.json')
p.update(last_verified_pid=None,last_verified_update=1000,last_remote_check_unix=final['time'],next_regular_check_not_before_unix=final['time']+660,
    active_scope='Improve predictive anonymous state representation after completed source-attention experiment',goal_status='active',completion_proven=False)
p['pending']=[
    'V23 and its v24 component audit are complete. No owned training or diagnostic remains active. Local exec sessions3054,79490,74196 all exited0. Do not restart terminal runs or duplicate completed development evaluations.',
    'V23dev500=5/80,1000=2/80; reference43/80 remains unmet. Saved training-branch dynamics improved47.594% to53.453%, but this is not closed-loop success. All evaluated actions retain exact proofs and neural boundary checks.',
    'V24 offline vocabulary-resolution probe provides a next concrete direction: learn a finer anonymous vocabulary with no semantic labels. Under identical clustering methodology,128 vs512 states gave deterministic recorded-label fit ceilings92.655% vs95.929%; four training layouts excluded from centroid fit gave91.483% vs95.449%. These are oracle fit bounds, not learned prediction accuracy or navigation performance.',
    'V24512centers are diagnostics/state_resolution_v24_centers_512.pt under the remote isolated root. Fitted on65536 frozen visual/history features from training layouts excluding112/113/120/121. All145188 source/successor features were used only for audit labels; no new rollout or policy update. The frozen feature encoder had already seen all original training layouts, so excluded-centroid figures are not an independent generalization evaluation.',
    'A plausible next bounded experiment is to distill these512 automatically learned definitions into a new full neural decoder and train complete goal-directed rules using the preserved raw input caches. Preserve original checkpoints. Account for16x larger dense transition table: reduce per-GPU batch or generate only needed rows during supervised dynamics training; still export full rules at boundaries and run the identical exact-goal interpreter. Do not replace the goal with a reachable state.',
    'Current v23 immutable1000 checkpoint SHA651dda94e6889b6d47a01b52cc0bf9ac9177ce8c80ade8e0522f4310fd3f0d36. Its source and trained weights are preserved; v21grounding/definition37 tensors remain unchanged. V24centroids have not been deployed.',
    'All8066 transition-bank boundary histories were recovered exactly; do not repeat completed recovery. Data replayed_boundary_features_v20/manifest.json,100%canonical395234patch IDs. Six banks have137122 action outcomes; demos24848frames2578episodes3680windows.',
    'V22actual saved32-training-task audit: first state matches43/43 programs;47/216 autonomous and73/216 one-step correct.23/36 executed plans diverge at first action. Wrong chosen transitions have meanconfidence0.771. This indicates physical dynamics mismatch despite formally correct proofs.',
    'Preserve all checkpoints, fixed train/dev/final partitions, and other GPU0/1/2/6 jobs. Only free3/4/5/7 are authorized. No reward/value table, semantic state, map, direct actor fallback, goal replacement or intermediate observation correction.',
    'Best exact-symbolic development result remains5/80 (v12step1600 andv23step500), reference43/80dev44/80fixed. Finalfixed80 untouched. Full success parity and complete execution-speed parity remain unproved; goal stays active.'
]
p['previous_turn_classification']='progress: completed two causal dynamics audits, implemented and trained source-evidence rule generation, evaluated two checkpoints, and identified a data-only finer-state direction; performance goal still unachieved'
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(goal_complete=False,latest_update=1000,development=[(r['update'],r['successes']) for r in final['validation']],owned_training_active=False)))
