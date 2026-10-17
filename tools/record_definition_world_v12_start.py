import json,time
from pathlib import Path,PurePosixPath
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules'
root=PurePosixPath('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
files={'diagnostics/proof_collection_v11_full_audit.json':'proof_collection_v11_full_audit.json',
    'diagnostics/v11_3200_own_path_fidelity.json':'v11_3200_own_path_fidelity.json',
    'initialization_definition_world_v12/report.json':'definition_world_v12_initialization.json',
    'diagnostics/definition_world_v12_preflight.json':'definition_world_v12_preflight.json'}
c=connect()
try:
    with c.open_sftp() as ftp:
        for src,dst in files.items():ftp.get(str(root/src),str(local/'evidence'/dst))
finally:c.close()
def read(name):return json.loads((local/'evidence'/name).read_text(encoding='utf-8-sig'))
p=json.loads((local/'goal_progress.json').read_text(encoding='utf-8'))
check=read('definition_world_v12_startup.json');collection=read('proof_collection_v11_final.json')
assert check['process_alive'] and check['process_identity_verified'] and check['committed_hash_verified']
assert check['status']['update']>0 and all(not r['alive'] and r['complete'] for r in collection['collection'])
full=read('proof_collection_v11_full_audit.json');assert full['full_collection_audited'] and full['valid_states']==420
p['active_collection'].update(state='complete',files=75,no_proof_skips=21,valid_states=420,
    full_collection_audited=True,processes=collection['collection'])
fidelity=read('v11_3200_own_path_fidelity.json')['results'][0]
assert fidelity['matching_collection_checkpoint'] and fidelity['executed_rule_autonomous_correct']==18
p['v11_3200_own_path_fidelity']={k:v for k,v in fidelity.items() if k!='rows'}
p['historical_vocabulary_rules_v11_best_development']=p.get('current_best_development')
p['historical_vocabulary_rules_v11_latest_development']=p.get('latest_completed_development')
key='definition_world_v12';run=str(root/('run_'+key));when=(local/'evidence/definition_world_v12_startup.json').stat().st_mtime
p.update(architecture='anonymous_definition_conditioned_world_v12',remote_run=run,remote_source=check['launcher']['source'],
    last_verified_update=check['status']['update'],last_verified_pid=check['status']['pid'],last_observed_state='training',
    last_remote_check_unix=check['time'],last_local_check_unix=when,
    next_regular_check_not_before_unix=check['time']+660,next_regular_local_check_not_before_unix=when+660,
    current_best_development=None,latest_completed_development=None,last_record_update_unix=time.time())
p['active_runs'][key]=dict(path=run,source=check['launcher']['source'],launcher=check['launcher'],
    latest_status=check['status'],latest_checkpoint=check['checkpoint'],process_alive=True,process_identity_verified=True,
    committed_hash_verified=True,recent=check['recent'],training_scope=check['training_scope'],validation=check['validation'])
p['definition_world_v12_initialization']=read('definition_world_v12_initialization.json')
p['definition_world_v12_preflight']={k:v for k,v in read('definition_world_v12_preflight.json').items() if k not in ['metrics','counterfactual']}
p['definition_world_v12_design']=dict(
    rule_inputs='learned source/destination definition vectors, action ID, boundary context',
    generated='complete128x17x128 transition probabilities and unchanged discrete goal interface',
    changed_from_v7='stable unsupervised vocabulary from v11, retained v11 goal and perception, transferred rule distributions and expanded six-bank actual transition data',
    initialization_limit='Only44.6% teacher rule argmax agreement; this is a new learned decoder, not a behavior-identical restart',
    losses='original joint dynamics/goal/proof objective; rejected optional loss interventions disabled',
    retained='full exact proof planning, no runtime observer correction or manual state/return tables')
p['pending']=[
    'After660 seconds inspect existing run_definition_world_v12 with check_predictive_training.py --run; never restart on timeout',
    'Read each independent development400/800/1200/1600/2000 once; compare navigation, no-proof failures, erroneous STOP and learned-state fidelity',
    'Use updated audit_grounded_rule_fidelity.py and diagnose_graph_components.py for architecturev12; runtime evaluator is evaluate_definition_world_v12.py',
    'Do not redeploy either rejected loss-only pilot; preserve the completed v11 checkpoint and all fixed splits',
    'Audit frozen representation at completion and actual full execution before any completion claim; current goal remains far from reference43/80 development44/80 final fixed',
    'No map, manually named state, reward/value table, collision/eligibility module, generated-goal replacement, direct neural actor or within-block observation correction']
p['current_turn_classification']='progress: v11 completed, exact runtime-state fidelity and all96 collection tasks audited, two loss candidates rejected, definition-conditioned v12 training actually updating'
(local/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(run=run,update=check['status']['update'],collection_audited_states=420,
    next_regular_check=when+660,goal_complete=False)))
