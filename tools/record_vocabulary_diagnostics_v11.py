import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(name):return json.loads((root/'evidence'/name).read_text(encoding='utf-8-sig'))
p=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
branch=read('vocabulary_v11_train_branch_ceiling.json')['results'][0]
p['vocabulary_v11_training_branch_ceiling']={k:v for k,v in branch.items() if k!='rows'}
p['vocabulary_v11_training_branch_ceiling']['interpretation']='Post-execution training-data label audit only. Deterministic ceiling3245/3621 is lower than the old representation3386/3621 on these same old paths; endpoint aliases improved but branch predictability did not universally improve.'
audit=read('v10_development_saved_navigation.json')
p['v10_saved_navigation_audit']={k:v for k,v in audit.items() if k!='rows'}
p['goal_context_capacity_v11']=read('goal_context_capacity_v11.json')
p['goal_context_capacity_v11']['interpretation']='Larger goal head fits training perfectly but does not improve internal held-out training labels; not deployed and not a navigation evaluation.'
p['current_turn_classification']='progress: completed saved-trace full-proof audit, measured new-state branch ambiguity, and rejected goal-head capacity-only explanation'
p['last_record_update_unix']=time.time()
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(recorded=True,regular_poll_gate_unchanged=p['next_regular_local_check_not_before_unix'],goal_complete=False)))
