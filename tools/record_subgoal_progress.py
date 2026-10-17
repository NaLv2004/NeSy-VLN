"""Keep the active implementation record distinct from retired experiments."""
import json,time
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'predictive_rules/goal_progress.json';d=json.loads(p.read_text())
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
for key in ['current_best_development','latest_completed_development','current_contract_tests','next_investigation']:
    if key in d:d['historical_'+key]=d.pop(key)
d.update(goal_status='active',completion_proven=False,remote_run=root+'/run_subgoal_v1',remote_source=root+'/src_subgoal_v1',
    architecture='anonymous_subgoal_rules_v1',last_verified_pid=514106,last_verified_update=None,
    last_observed_state='launched_not_yet_confirmed_optimizer_update',train_gpus=[3,5],evaluation_gpu=7,
    primary_gpu=3,generated_parameters=['definitions','parents','transitions','subgoal_logits'],
    contract_tests=dict(passed=7,total=7),real_training_data_preflight_passed=True)
d['invariants']=[
    'No manually defined semantic states, collision/outcome classifiers, reward/value tables or direct action heads',
    'Network generates anonymous definitions, all transition parameters, and an unmodified discrete subgoal',
    'Only exact goal-reaching rule derivations may execute; no approximate or neural fallback',
    'No neural forward, observation correction, feedback input or maps between ten-action revisions',
    'Future images supervise learned states only; never enter the program generator',
    'Development and fixed validation layouts remain excluded from training',
    'Only authorized GPUs 3,4,5,7; preserve unrelated processes and saved checkpoints']
for name,step,digest in [('anonymous_v5',2800,'529951339ee2c11d49c4078d88f7a066d81bda00cea6ca9b3746d2b444314e7a'),('anonymous_v6',800,'6672c7f94c84176db4bd5cd4c55c7eb51ba58e02a6608690800c98c4049b2b52')]:
    r=d['active_runs'].setdefault(name,{})
    r.update(last_observed_state='retired_user_rejected_reward_value_tables',latest_preserved_update=step,
        retired_checkpoint_sha256=digest,superseded=True)
d['active_runs']['subgoal_v1']=dict(pid=514106,train_gpus=[3,5],evaluation_gpu=7,path=root+'/run_subgoal_v1',state='launched')
d['pending']=['Confirm distributed optimizer updates and committed checkpoint',
    'Measure exact symbolic plan reachability and actual development80 navigation success',
    'Audit all executed validation actions and neural boundary isolation',
    'Improve until development43/80 and independent fixed44/80 reference parity, plus computation timing, are demonstrated']
d['current_contract_tests']=dict(passed=7,total=7,scope='new subgoal schema, exact rule proof, no neural fallback, gradients and observation isolation')
d['last_record_update_unix']=time.time();p.write_text(json.dumps(d,indent=2),encoding='utf-8')
