import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(name):return json.loads((root/'evidence'/name).read_text(encoding='utf-8-sig'))
p=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
check=read('proof_collection_v11_startup.json')
assert all(r['alive'] and r['identity_verified'] for r in check['collection'])
assert check['files']>0
when=(root/'evidence/proof_collection_v11_startup.json').stat().st_mtime
p['previous_collection_v9']=p.get('active_collection')
p['active_collection']=dict(run='counterfactual_proof_v11',path='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/counterfactual_proof_v11',
    source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_proof_collection_v11',
    checkpoint='run_vocabulary_rules_v11/checkpoint_003200.pt',training_tasks=96,
    manifest_sha256='b2b14b718b6c287b5bdf5eac47e6c067b71ca5cc8f99ccedbb1cd31aa3e9fb29',
    state='collecting',files=check['files'],no_proof_skips=check['no_proof_skips'],processes=check['collection'],
    last_local_check_unix=when,next_regular_local_check_not_before_unix=when+660,
    last_remote_check_unix=check['time'],next_regular_remote_check_not_before_unix=check['time']+660,
    purpose='Audit learned dynamics on newly executed exact proofs and retain training-only counterfactual labels')
for key,file in [('v11_2000_saved_navigation','v11_2000_development_saved_navigation.json'),
    ('v11_2000_train_components','v11_2000_train_components.json'),('goal_rule_gradient_conflict_v11_2000','goal_rule_gradient_conflict_v11_2000.json')]:
    p[key]={k:v for k,v in read(file).items() if k!='rows'}
fidelity=read('v11_2000_training_fidelity.json')['results'][0]
p['v11_2000_training_fidelity']={k:v for k,v in fidelity.items() if k!='rows'}
p['trainable_goal_context_probe']={k:v for k,v in read('trainable_goal_context_v11.json').items() if k!='history'}
p['trainable_goal_context_probe']['decision']='Not deployed; best internal holdout34.3% is modest and final29.9% shows overfit. It does not establish navigation improvement.'
p['active_loss_pilot']=dict(path='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/pilot_world_objective_v11',
    source='src_world_objective_pilot_v11',source_checkpoint='run_vocabulary_rules_v11/checkpoint_002000.pt',
    updates=300,device='cpu',threads=2,loss_intervention='proof_action_imitation weight1 to0; all other terms unchanged',
    exec_session=44187,pid=599677,state='running_last_verified',not_deployed=True,
    rationale='Measured rule gradients from proof imitation are larger than world-model gradients; wrong generated goals can conflict with demonstrated target states. This pilot tests the effect rather than claiming causality.')
p['pending']=[
    'Continue waiting on existing exec session44187 for the CPU300-update loss pilot; no duplicate launch on timeout',
    'When candidate.pt is committed, compare training-only rule fidelity and proof components against step2000 before deciding whether one development evaluation on GPU7 is warranted',
    'After the collection660-second gate inspect existing PIDs600917/600918/600919, then full-audit every collected file and no-proof case and compute rule fidelity',
    'V11 is terminal at3200; final development4/80 with no boundary violations, frozen representation verified. Do not extend this unchanged run or re-evaluate existing checkpoints',
    'Keep final fixed80 isolated; target reference43/80 development and44/80 fixed plus complete inference speed parity remain unmet',
    'No reward/value table, manual semantic states, map, goal substitution, direct actor or within-block observation corrections']
p['current_turn_classification']='progress: v11 completed and measured4/80, full saved proofs rederived, world-model/plan-gradient interference measured, bounded loss pilot and training-only own-path collection launched'
p['last_record_update_unix']=time.time()
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(collection_tasks=96,collection_verified_files=check['files'],next_collection_check=when+660,goal_complete=False)))
