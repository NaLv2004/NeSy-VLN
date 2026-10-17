import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(name):return json.loads((root/'evidence'/name).read_text(encoding='utf-8-sig'))
p=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
p['active_loss_pilot'].update(state='completed_not_promoted',exec_session_closed=True,
    completion=read('v11_world_pilot_complete.json'),decision='First proof action agreement fell41/128 to18/128 and true autonomous transitions31/213 to25/213; no development evaluation or production restart')
for key,prefix in [('world_objective_pilot','v11_world_pilot'),('goal_compatible_pilot','v11_goal_compatible')]:
    components=read(prefix+'_train_components.json');fidelity=read(prefix+'_training_fidelity.json')['results'][0]
    p[key]=dict(components={k:v for k,v in components.items() if k!='rows'},
        fidelity={k:v for k,v in fidelity.items() if k!='rows'},completion=read(prefix+'_complete.json'),
        promoted=False,development_evaluation_run=False)
p['goal_compatible_pilot'].update(
    source='src_goal_compatible_pilot_v11',checkpoint='pilot_goal_compatible_v11/candidate.pt',
    change='Only mask proof-imitation loss when generated goal differs from demonstrated endpoint; no teacher goal is used inside the proof',
    contract_test_passed=True,decision='First action27/128 and autonomous17/213 were below source41/128 and31/213; rejected without development evaluation')
p['v11_3200_saved_navigation']={k:v for k,v in read('v11_3200_development_saved_navigation.json').items() if k!='rows'}
p['pending']=[
    'After the active collection660-second gate inspect existing PIDs600917/600918/600919; do not restart on a transport timeout',
    'Once all collection handles are terminal, full-audit96 tasks including NO_PROOF cases, then compare v11 step3200 inferred states with post-execution labels on its newly visited training states',
    'Both loss-only pilots are completed and rejected; do not deploy either optional loss argument or re-evaluate them on development',
    'Use own-path fidelity and observed STOP transition errors to choose the next representation/rule-decoder intervention; data aggregation alone and bigger goal heads have not solved navigation',
    'Preserve all saved checkpoints and fixed train/development/final splits. Latest regular run terminal3200; final development4/80; baseline43/80 development and44/80 fixed still unmet',
    'No reward/value table, manually named state, collision/eligibility head, map, generated-goal substitution, direct actor or within-block observation correction']
p['current_turn_classification']='progress: completed v11 and full saved-proof audits, bounded goal-head/encoder probes, measured gradient interference, rejected two loss interventions and started own-policy data collection'
p['last_record_update_unix']=time.time()
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(pilots_completed=True,pilots_promoted=0,goal_complete=False)))
