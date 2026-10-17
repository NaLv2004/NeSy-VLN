import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
path=root/'evidence/grounded_rules_v9_startup.json';check=read(path);p=read(root/'goal_progress.json')
assert check['process_alive'] and check['process_identity_verified'] and check['committed_hash_verified']
p['historical_rollout_proof_repaired_final_development']=p['latest_completed_development']
p['active_runs']['rollout_proof_v8r1'].update(last_observed_state='training_budget_complete',superseded=True,process_alive=False)
remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
p.update(remote_run=remote+'/run_grounded_rules_v9',remote_source=remote+'/src_grounded_rules_v9',
    last_verified_update=check['status']['update'],last_verified_pid=check['status']['pid'],last_observed_state=check['status']['state'],
    last_local_check_unix=path.stat().st_mtime,next_regular_local_check_not_before_unix=path.stat().st_mtime+660,
    last_remote_check_unix=check['time'],next_regular_check_not_before_unix=check['time']+660,
    current_best_development=None,latest_completed_development=None,last_record_update_unix=time.time())
p['active_runs']['grounded_rules_v9']=dict(path=p['remote_run'],**check['launcher'],latest_status=check['status'],
    latest_checkpoint=check['checkpoint'],process_alive=True,process_identity_verified=True,committed_hash_verified=True,
    recent=check['recent'],validation=check['validation'],training_scope=check['training_scope'])
p['bounded_trial']['state']='complete_rejected';p['bounded_trial']['promoted']=False
p['state_separation_trials']=dict(source_first_action_correct=61,samples=128,
    overlap=dict(first_action_correct=34,conflicts=150,nonterminal_STOP=4,promoted=False,sha256='4fcaf1cfd8ad6e83237c2b34853380e3b86baa9dc7c4260d3b5b17bd7321a024'),
    mutual_information=dict(first_action_correct=26,conflicts=142,nonterminal_STOP=6,promoted=False,sha256='9dc2081f6ee1946914b57ace5bd85a38243348f81d55ff7d71b010c1886e5aaf'),
    reason='Neither preserved action quality or sufficiently removed contradictory discrete successors; no development/final evaluation spent on these candidates')
collection=read(root/'evidence/proof_collection_v8r1_complete.json')
assert all(x['complete'] and not x['alive'] for x in collection['collection'])
p['new_transition_collection']=dict(path=remote+'/counterfactual_proof_v8r1',complete=True,selected_training_episodes=192,
    files=118,unreachable_no_action_skips=74,valid_states=506,action_branches=506*17,
    scalar_reward_fields=False,semantic_state_labels=False,validation_used=False,first16_files_audited=True)
p['actual_rule_prediction_audit']={k:v for k,v in read(root/'evidence/executed_rule_predictions_v8r1.json').items() if k!='rows'}
p['grounded_rules_preflight']={k:v for k,v in read(root/'evidence/grounded_rules_preflight.json').items() if k not in ['metrics','counterfactual']}
p['pending']=['Read v9 development400/800/1200 and frozen-representation verification after regular polling gate; never rerun the same checkpoint evaluation',
    'Compare actual executed state-transition accuracy on collected training-only paths to the frozen v8r1 baseline12/184; do not equate it with task success',
    'If correcting rule fitting still fails, revisit state representation capacity and deterministic abstraction rather than extending unchanged training',
    'Reference43/80 development and44/80 fixed success plus full inference speed remain unproven']
p['current_turn_classification']='progress: completed v8r1 validation; rejected two representation-loss pilots; collected and audited192 training episodes without feedback correction; measured12/184 actual successor agreement; launched and verified v9 rule/subgoal-only updates with fixed learned grounding'
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(update=p['last_verified_update'],next_check=p['next_regular_local_check_not_before_unix'],goal_complete=False)))
