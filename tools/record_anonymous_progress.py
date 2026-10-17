"""Record verified artifacts, keeping the full unmet goal and exact next checks."""
import json
from pathlib import Path
local=Path(__file__).resolve().parents[1]/'predictive_rules'
path=local/'goal_progress.json';p=json.loads(path.read_text())
snapshot=json.loads((local/'results/anonymous_snapshot.json').read_text())
manifest=json.loads((local/'results/anonymous_manifest.json').read_text())
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
p.update(goal_status='active',completion_proven=False,remote_run=root+'/run_anonymous_v5',
    remote_source=root+'/src_anonymous_v5',authorized_gpus=[3,4,5,7],primary_gpu=3,
    last_remote_check_unix=snapshot['time'],next_regular_check_not_before_unix=snapshot['time']+610)
p.pop('authorized_gpu',None)
p['invariants']=[
    'No manually defined semantic states, named semantic classifier tables or dedicated semantic supervision',
    'Internal categories and transition rules are anonymous and learned; scalar task score is external supervision only',
    'No neural forward, observation correction, feedback input or maps between ten-action revisions',
    'Every executable rule parameter is generated in full; actions come from multi-step rule search',
    'Development and fixed validation layouts remain excluded from training',
    'Only authorized GPUs 3,4,5,7; preserve unrelated processes and all committed checkpoints']
if 'best_new_development' in p:
    p['legacy_best_not_qualifying_under_clarified_contract']=p.pop('best_new_development')
current=snapshot['runs']['run_anonymous_v5'];status=current['status.json']
p['current_best_development']=current.get('best_development.json')
if current.get('validations'):
    p['latest_completed_development']=dict(run='anonymous_v5',**current['validations'][-1])
p['active_runs']['anonymous_v5']=dict(path=root+'/run_anonymous_v5',pid=current['launcher.json']['pid'],
    gpu=3,last_observed_update=status['update'],last_observed_state=status['state'],
    last_update_seconds=status.get('seconds'),architecture=manifest['architecture'],
    trainable_parameters=manifest['trainable_parameters'],generated_parameters=manifest['generated_parameters'])
for key in ['memory_v2','outcomes_v3']:
    row=snapshot['runs']['run_'+key]['status.json']
    p['active_runs'][key].update(last_observed_update=row['update'],last_observed_state=row['state'],superseded=True)
p['active_runs']['v1'].update(last_observed_state='retired_superseded',latest_preserved_update=5100,superseded=True)
p['last_verified_pid']=current['launcher.json']['pid'];p['last_verified_update']=status['update']
p['current_contract_tests']=dict(passed=7,total=7,scope='anonymous model, full objectives, multistep planning and neural isolation')
p['own_policy_collection']=dict(path=root+'/counterfactual_own_policy_v1',target=128,completed_files=snapshot['own_policy_files'],
    gpu4_pid=489203,gpu5_pid=489237,training_only=True,source_policy='frozen older checkpoint for coverage; not the new architecture')
p['own_policy_collection']['state']='complete' if snapshot['own_policy_files']==128 else 'collecting'
diagnosis_file=local/'results/anonymous_step800_diagnosis.json'
if diagnosis_file.exists():
    diagnosis=json.loads(diagnosis_file.read_text())
    audit=json.loads((local/'results/anonymous_step800_audit.json').read_text())
    if 'predicted_collision_mean' in p.get('diagnosis',{}):
        p['legacy_semantic_diagnosis']=p.pop('diagnosis')
    p['diagnosis']=dict(run='anonymous_v5',update=800,**{k:v for k,v in diagnosis.items() if k!='evaluation_summary'})
    p['independent_rollout_audit']=dict(run='anonymous_v5',update=800,**{k:v for k,v in audit.items() if k!='records'})
p['planning_alignment'].update(pilot_state='completed on legacy model; did not improve unseen-training first-action agreement',
    deployed_to_training=True,deployed_architecture='anonymous_v5',new_scalar_return_objective=True)
p['next_investigation']=[
    'Diagnose generic return overestimation: step800 predicted stop score +0.742 versus actual -0.744; 34/39 stops failed',
    'Measure anonymous state prediction errors under the same boundary-generated definitions on training-only transitions',
    'Investigate terminal-horizon value bootstrapping without adding hand-defined semantic states',
    'Compare navigation success against 43/80 dev and later 44/80 fixed; do not use supervised loss as success',
    'Measure actual rule-planning and boundary-neural runtime against the reactive reference',
    'Continue improvement; the full requested outcome remains unachieved',
    'Do not resume retired semantic-classifier experiments or launch the unstarted v4 trainer',
    'Stable status polls remain at least 600 seconds apart']
path.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(run=p['remote_run'],update=status['update'],next_check=p['next_regular_check_not_before_unix'],goal_complete=False)))
