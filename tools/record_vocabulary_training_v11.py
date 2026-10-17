"""Record actual startup and completed vocabulary evidence, without polling training."""
import json,time
from pathlib import Path,PurePosixPath
from predictive_remote import connect

local=Path(__file__).resolve().parents[1]/'predictive_rules'
root=PurePosixPath('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
files={
    'diagnostics/subgoal_alias_corpus_v9.json':'subgoal_alias_corpus_v9.json',
    'diagnostics/alias_geometry_v9.json':'alias_geometry_v9.json',
    'diagnostics/unsupervised_vocabulary_v9/report.json':'unsupervised_vocabulary_v9.json',
    'initialization_vocabulary_v11r1/report.json':'vocabulary_initialization_v11r1.json',
    'diagnostics/vocabulary_rules_preflight.json':'vocabulary_rules_preflight.json',
}
c=connect()
try:
    with c.open_sftp() as ftp:
        for src,dst in files.items():ftp.get(str(root/src),str(local/'evidence'/dst))
finally:c.close()
def read(name):return json.loads((local/'evidence'/name).read_text(encoding='utf-8-sig'))
startup=read('vocabulary_rules_v11_startup.json');initial=read('vocabulary_initialization_v11r1.json')
assert startup['process_alive'] and startup['process_identity_verified'] and startup['committed_hash_verified']
assert startup['status']['update']>0 and initial['passed']
p=json.loads((local/'goal_progress.json').read_text(encoding='utf-8'))
p['historical_grounded_aggregate_v10_best_development']=p.get('current_best_development')
p['historical_grounded_aggregate_v10_latest_development']=p.get('latest_completed_development')
key='vocabulary_rules_v11';run=str(root/('run_'+key))
checked=(local/'evidence/vocabulary_rules_v11_startup.json').stat().st_mtime
p.update(remote_run=run,remote_source=startup['launcher']['source'],last_verified_update=startup['status']['update'],
    last_verified_pid=startup['status']['pid'],last_observed_state='training',
    last_remote_check_unix=startup['time'],last_local_check_unix=checked,
    next_regular_check_not_before_unix=startup['time']+660,next_regular_local_check_not_before_unix=checked+660,
    current_best_development=None,latest_completed_development=None,
    current_turn_classification='progress: full corpus alias audit, unsupervised visual vocabulary and neural distillation, verified three-GPU training updates',
    last_record_update_unix=time.time())
p['active_runs'][key]=dict(path=run,source=startup['launcher']['source'],launcher=startup['launcher'],
    latest_status=startup['status'],latest_checkpoint=startup['checkpoint'],process_alive=True,
    process_identity_verified=True,committed_hash_verified=True,recent=startup['recent'],
    training_scope=startup['training_scope'],validation=startup['validation'])
p['vocabulary_initialization_v11r1']=initial
p['vocabulary_rules_preflight']={k:v for k,v in read('vocabulary_rules_preflight.json').items() if k not in ['metrics','counterfactual']}
p['unsupervised_vocabulary_evidence']=dict(
    files=list(files.values()),fit_only_training_images=True,semantic_labels_used=False,
    fitting_scope='80% of training episode IDs, with remaining training IDs for an internal diagnostic; neither development nor fixed validation used',
    diagnosis='Original definitions alias changed images; learned visual centers reduce endpoint aliases, not yet a navigation-success result',
    runtime='Complete neural definition decoder; no external clustering file, manual state names, action fallback, reward/value table or within-block image correction',
    failed_initialization_preserved='initialization_vocabulary_v11; distillation assignment acceptance failed, not deployed')
p['pending']=[
    'Inspect run_vocabulary_rules_v11 after the 660-second gate with the generic checker --run; never restart because of transport timeout',
    'Evaluate development80 once per committed 400 updates through 3200; do not repeat checkpoints or use final fixed80 for tuning',
    'Audit actual executed rule fidelity and STOP behavior, not only losses or alias reduction; compare with v10 final1/80',
    'Verify representation bitwise unchanged after rules/goal training; preserve all initialization and training checkpoints',
    'No manual semantic state, collision/eligibility head, reward/value table, map, unreachable-goal replacement or per-action neural feedback',
    'Goal remains unmet: reference43/80 development and44/80 final fixed plus complete inference-speed parity are not achieved']
(local/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(run=run,update=startup['status']['update'],initialization_passed=True,
    mean_update_seconds=startup['recent']['mean_seconds'],next_regular_local_check=checked+660,goal_complete=False)))
