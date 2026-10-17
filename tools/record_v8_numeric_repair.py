import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
path=root/'evidence/rollout_proof_v8r1_startup.json';check=read(path);p=read(root/'goal_progress.json')
assert check['process_alive'] and check['process_identity_verified'] and check['committed_hash_verified']
assert check['status']['update']>1413
old=p['active_runs']['rollout_proof_v8'];old.update(last_observed_state='failed_nonfinite_gradient',process_alive=False,
    attempted_failed_update=1413,last_preserved_update=1400,superseded=True,
    failure_reproduced_from_optimizer_and_rng=True)
p['historical_rollout_proof_best_development']=p['current_best_development']
p['historical_rollout_proof_latest_development']=p['latest_completed_development']
remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
p.update(remote_run=remote+'/run_rollout_proof_v8r1',remote_source=remote+'/src_rollout_proof_v8r1',
    last_verified_update=check['status']['update'],last_verified_pid=check['status']['pid'],last_observed_state=check['status']['state'],
    last_local_check_unix=path.stat().st_mtime,next_regular_local_check_not_before_unix=path.stat().st_mtime+660,
    last_remote_check_unix=check['time'],next_regular_check_not_before_unix=check['time']+660,
    current_best_development=None,latest_completed_development=None,last_record_update_unix=time.time())
p['active_runs']['rollout_proof_v8r1']=dict(path=p['remote_run'],**check['launcher'],latest_status=check['status'],
    latest_checkpoint=check['checkpoint'],process_alive=True,process_identity_verified=True,committed_hash_verified=True,
    recent=check['recent'],validation=check['validation'])
p['v8_numeric_repair']=dict(failed_attempt=1413,restored_update=1400,
    source_checkpoint_sha256='44af63bcc9275b5538e17a390ac35ff843b38c2eed22b1bcd9c51646ef1930b0',
    reproduced_exactly=True,offending_term='teacher_path_goal',offending_parameter='history.weight_ih_l0',
    observed_recurrent_dtype_under_bf16_autocast='torch.float16',repaired_recurrent_dtype='torch.float32',
    original_failing_batch_repaired_all_gradients_finite=True,optimizer_and_per_rank_rng_restored=True,
    new_neural_inference_between_revisions=False,loss_terms_removed=False)
audit=read(root/'evidence/v8step1200_navigation_proofs.json')
p['v8_actual_failure_audit']={k:v for k,v in audit.items() if k!='rows'}
components=read(root/'evidence/components_v6step400_goal_semantics.json')
p['subgoal_stop_diagnosis']=dict(source='v6 step400, initial weights for v8',training_only=True,
    nonterminal_windows=components['nonterminal_windows'],premature_stopping_plans=components['premature_stop_on_nonterminal_windows'],
    interpretation='Premature stopping exists on demonstrations; intermediate-goal/terminal-goal ambiguity is a hypothesis alongside state aliasing, not yet isolated causal proof')
p['pending']=['Read v8r1 development1600/2000 after stable poll gate; do not duplicate evaluations',
    'Investigate unsupported STOP transitions, intermediate versus final goals and anonymous-state aliasing without reward tables or semantic hand labels',
    'Reach reference43/80 development and44/80 fixed, then verify full inference speed; objective remains unmet']
p['current_turn_classification']='progress: independently audited v8 trajectories, exactly reproduced numerical exit, isolated hidden FP16 recurrent overflow, repaired it with the same failing batch and restored committed optimizer/RNG; verified real updates beyond the failed step'
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(update=p['last_verified_update'],next_check=p['next_regular_local_check_not_before_unix'],goal_complete=False)))
