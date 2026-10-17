"""Record completed v9 evidence and the verified next collection handles."""
import json,time
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(name):return json.loads((r/'evidence'/name).read_text(encoding='utf-8-sig'))
p=json.loads((r/'goal_progress.json').read_text(encoding='utf-8'))
status=read('grounded_rules_v9_final_check.json')
assert status['status']['state']=='training_budget_complete' and not status['process_alive']
assert status['committed_hash_verified'] and read('grounded_rules_v9_frozen_verified.json')['passed']
fidelity=read('grounded_fidelity_v9.json')
p['grounded_rules_v9_completed']=dict(
    update=1200,frozen_representation_verified=True,
    best_development=max(status['validation'],key=lambda x:x['successes']),
    final_development=status['validation'][-1],
    training_fidelity=[{k:v for k,v in row.items() if k!='rows'} for row in fidelity['results']],
    training_components={k:v for k,v in read('components_grounded_v9_1200.json').items() if k!='rows'},
    navigation_parity=False,interpretation='Rule fitting improved on old collected training trajectories, but closed-loop development success did not improve')
startup=read('proof_collection_v9_startup.json')
assert all(row['alive'] and row['identity_verified'] and row['status']['state']=='collecting' for row in startup['collection'])
p['active_collection']=dict(path='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/counterfactual_proof_v9',
    source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_proof_collection_v9',
    selected_training_episodes=384,files_at_last_check=startup['files'],no_proof_skips_at_last_check=startup['no_proof_skips'],
    processes=[{k:v for k,v in row.items() if k!='log_tail'} for row in startup['collection']],
    source_update=1200,source_run='grounded_rules_v9',same_as_evaluation_single_frame_encoding=True,
    no_observation_correction=True,training_only=True,old_banks_preserved=True)
local=(r/'evidence/proof_collection_v9_startup.json').stat().st_mtime
p.update(last_remote_check_unix=startup['time'],last_local_check_unix=local,
    next_regular_local_check_not_before_unix=local+660,next_regular_check_not_before_unix=startup['time']+660,
    last_record_update_unix=time.time(),current_turn_classification='progress: v9 completed with verified fixed representation, measured world-model fitting vs navigation gap, started four-card training-only on-policy branch collection')
p['pending']=[
    'After the 660-second gate, inspect the four existing proof_collection_v9 handles; never restart on timeout',
    'Audit completed on-policy branches and combine with earlier training-only transition banks to avoid replacing broad coverage with a narrow recent bank',
    'Test whether broader learned transition coverage improves complete executed proofs; do not claim that old-trajectory fitting equals new-trajectory navigation',
    'Retain generated subgoals and all exact proof constraints, no hand-defined states, rewards, terminal classifiers or neural correction',
    'Success reference43/80 development and44/80 fixed plus full end-to-end inference-speed parity remain unmet; fixed80 remains untouched']
(r/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(collection='started_and_verified',gpus=[3,4,5,7],files=startup['files'],
    last_training_update=1200,latest_development=status['validation'][-1]['successes'],goal_complete=False,next_check=local+660)))
