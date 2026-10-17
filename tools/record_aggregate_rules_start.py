import json,time
from pathlib import Path,PurePosixPath
r=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(name):return json.loads((r/'evidence'/name).read_text(encoding='utf-8-sig'))
p=json.loads((r/'goal_progress.json').read_text(encoding='utf-8'))
startup=read('aggregate_rules_v10_startup.json');collection=read('proof_collection_v9_after_gate.json')
assert startup['process_alive'] and startup['process_identity_verified'] and startup['committed_hash_verified']
assert startup['status']['update']>0 and all(not x['alive'] and x['complete'] for x in collection['collection'])
full=read('proof_collection_v9_full_audit.json');assert full['full_collection_audited'] and full['valid_states']==1593
preflight=read('aggregate_rules_preflight.json');assert preflight['passed']
run='grounded_aggregate_v10';path=PurePosixPath(startup['launcher']['source']).parent/('run_'+run)
local=(r/'evidence/aggregate_rules_v10_startup.json').stat().st_mtime
p['active_collection'].update(state='complete',files=collection['files'],unreachable_no_action_skips=collection['no_proof_skips'],
    valid_states=1593,full_collection_audited=True,processes=collection['collection'])
p['historical_grounded_rules_v9_best_development']=p.get('current_best_development')
p['historical_grounded_rules_v9_latest_development']=p.get('latest_completed_development')
p.update(remote_run=str(path),remote_source=startup['launcher']['source'],
    last_verified_update=startup['status']['update'],last_verified_pid=startup['status']['pid'],last_observed_state='training',
    last_remote_check_unix=startup['time'],last_local_check_unix=local,
    next_regular_check_not_before_unix=startup['time']+660,next_regular_local_check_not_before_unix=local+660,
    current_best_development=None,latest_completed_development=None,
    current_turn_classification='progress: full new collection audited, older transitions audited and aggregated, actual three-card updates verified',
    last_record_update_unix=time.time())
p['active_runs'][run]=dict(path=str(path),source=startup['launcher']['source'],
    launcher=startup['launcher'],latest_status=startup['status'],latest_checkpoint=startup['checkpoint'],
    process_alive=True,process_identity_verified=True,committed_hash_verified=True,recent=startup['recent'],
    training_scope=startup['training_scope'],validation=startup['validation'])
p['aggregate_rules_preflight']={k:v for k,v in preflight.items() if k not in ['metrics','counterfactual']}
p['symbol_precision_audit']={k:v for k,v in read('symbol_precision_v9.json').items() if k!='rows'}
p['symbol_precision_audit']['interpretation']='Small training/inference mismatch; not sufficient to explain poor navigation, no numerical redesign adopted'
p['reconstruction_state_use']=read('reconstruction_state_use_v9.json')
p['reconstruction_state_use']['interpretation']='Both state and context affect reconstruction; evidence does not support claiming that states are entirely ignored'
p['collector_encoding_correction']=dict(evidence='evidence/collector_v8r1_environment.py',
    correction='Prior eager encoder already loops single frames; CUDA graph collection is an optimization, not a repair of batched-convolution semantics')
p['grounded_rules_v9_fidelity_own_paths']={k:v for k,v in read('grounded_fidelity_v9_on_own_paths.json')['results'][0].items() if k!='rows'}
p['pending']=[
    'After the 660-second gate, inspect run_grounded_aggregate_v10 via check_predictive_training.py; use the existing launcher, never restart on timeout',
    'Read development400/800/1200 once each and compare actual navigation; verify frozen representation at completion',
    'Compare aggregated-model dynamics on the old and newly executed training-only paths, separating fitting from closed-loop success',
    'If broader transition coverage still fails navigation, revisit goal/state abstraction using the measured 27/80 initial-goal aliases; do not prolong an unchanged run',
    'Preserve no manual states, reward/value tables, maps, direct neural action fallback, or within-block observation correction',
    'Reference43/80 development and44/80 fixed and full inference-speed parity remain unmet; final fixed80 remains excluded from selection']
(r/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(run=run,update=p['last_verified_update'],transition_states=sum(b['states'] for b in preflight['banks']),
    mean_update_seconds=startup['recent']['mean_seconds'],next_check=local+660,goal_complete=False)))
