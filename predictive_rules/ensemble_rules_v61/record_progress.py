import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
result=read('result.json');obs=read('evaluation_observation.json')
held=result['history'][-1]['results']['holdout']
record=dict(phase=obs['status']['state'],source=result['source'],
    hypothesis='Better calibrated action-outcome heads plus rectangular robust categorical reachability may reduce exploitation of optimistic transitions.',
    training_pilot=result,export_and_runtime_observation=obs,
    planner_contract=read('planner_contract.json'),premise_audit=read('premise_audit.json'),
    calibration_summary={name:held[name] for name in ('baseline','single_control','ensemble')},
    caveat='Head ensemble improves Brier/ECE but not changed-state accuracy; ensemble NLL is slightly worse than single control. No navigation improvement claimed without full80 results.',
    last_remote_check_unix=obs['time'],next_regular_check_not_before_unix=obs['time']+630,
    main_training_interrupted=False,goal_status='active',completion_proven=False,
    runtime=dict(neural_planner=False,map=False,handcrafted_semantic_states=False,
        action_selection='NumPy max-action/min-member finite-horizon goal reachability',
        generated_goal_unchanged=True,memory_grounding_only_at_renewal=True,min_renewal_actions=5,max_actions=32),
    full_dev80_used_for_pilot=False,full_dev80_evaluation_started=obs['status']['state'] in ('evaluating_full80','complete'),
    finalfixed80_used=False,
    references=['https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf',
        'https://pubsonline.informs.org/doi/abs/10.1287/moor.1040.0129?journalCode=moor'],
    source_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
if (HERE/'exit_event.json').exists():
    record['coordinator_exit_event']=read('exit_event.json')
if obs['status']['state']=='complete':
    assert obs['status']['exit_codes']==[0,0] and obs['summary']['episodes']==80
    assert not any(p['command'] for p in obs['processes'])
    record['decision']='complete_not_promoted'
    record['reason']='Full80 success24 versus source26 and best v60=27;10 gained,12 lost. Reduced repetition did not improve full-set success.'
    record['next_regular_check_not_before_unix']=None
(HERE/'experiment_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
goal_path=HERE.parent/'goal_progress.json'
goal=json.loads(goal_path.read_text(encoding='utf-8'))
goal.update(phase='robust_symbolic_evaluation_'+obs['status']['state'],
    independent_experiment=str(HERE/'experiment_record.json'),
    active_evaluation_coordinator_pid=obs['launch']['pid'],
    active_evaluation_worker_pids=obs['status'].get('workers',[]),
    last_evaluation_check_unix=obs['time'],next_evaluation_check_not_before_unix=obs['time']+630,
    next_action='The v60 training is finished and retained. Check the exact v61 evaluation coordinator/worker handles after the630-second gate or its already registered exit event. Read the complete original80-task result before promoting any planner; do not repeat completed evaluations. finalfixed80 remains untouched.',
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    goal_status='active',completion_proven=False)
if obs['status']['state']=='complete':
    goal.update(active_evaluation_coordinator_pid=None,active_evaluation_worker_pids=[],
        next_evaluation_check_not_before_unix=None,
        latest_verified_full80=dict(run='ensemble_rules_v61',update=400,successes=obs['summary']['successes'],episodes=80),
        next_action='v61 completed and was not promoted. Inspect the exact v62 CPU pilot2197424. Compare its predetermined final1200 fixed-label results with the matched control before any full80 navigation trial. Preserve v60 best27/80 and finalfixed80.')
goal_path.write_text(json.dumps(goal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(phase=record['phase'],processes=obs['processes'],next_check=obs['time']+630,goal_complete=False)))
