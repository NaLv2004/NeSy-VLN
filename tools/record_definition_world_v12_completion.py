import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
p=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
c=json.loads((root/'evidence/definition_world_v12_final_check.json').read_text(encoding='utf-8-sig'))
assert not c['process_alive'] and c['status']['state']=='training_budget_complete'
assert c['status']['update']==2000 and c['committed_hash_verified']
assert c['frozen_representation_verified']['passed'] and len(c['validation'])==5
p['active_runs']['definition_world_v12'].update(frozen_representation_verified=c['frozen_representation_verified'],
    latest_preserved_update=2000,state='completed_not_extended')
p['pending']=[
    'v12 is terminal and all five development evaluations are already recorded; do not poll/restart/extend it unchanged.',
    'Preserve v12 best step1600 (5/80) and final2000 (2/80), v11 checkpoints, and all fixed splits. Reference remains43/80 development44/80 fixed.',
    'Both supervised-world probes are rejected, not deployed; v13 context probe did not establish generalization. No additional navigation reruns for those weak candidates.',
    'Next work must address action-relevant learned-state abstraction and transition generalization using training-only evidence; simply fitting existing labels longer or enlarging the context head already overfits.',
    'Any next architecture must preserve full neural generation of definitions/transitions/discrete subgoal, exact complete symbolic proofs, no maps/manual semantic states/value or reward table/direct actor/within-block observation correction.',
    'Only use free authorized GPUs3/4/5/7 after a launch-time availability check. Other processes0/1/2/6 stay untouched.',
    'Navigation parity and full end-to-end speed parity remain unproven; do not mark the goal complete from loss/contract tests alone.'
]
p['current_turn_classification']='progress: v12 completion verified; state-error geometry and two bounded world-fitting experiments establish persistent transition generalization failure; neither rejected candidate was promoted'
p['last_record_update_unix']=time.time()
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(completed_updates=2000,best=[5,80],final=[2,80],frozen_verified=True,goal_complete=False)))
