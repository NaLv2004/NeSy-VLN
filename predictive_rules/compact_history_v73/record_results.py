import json,time
from pathlib import Path
folder=Path(__file__).parent;parent=folder.parent
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
current=read(folder/'completion_observation.json')
for name in ('normal_near_medium_v67','goal_probability_v68'):
    directory=parent/name;record=read(directory/'experiment_record.json');result=read(directory/'result.json')
    record.update(state='complete',result=result,goal_complete=False)
    record['decision']='Reject navigation37/80 versus source39/80' if name.endswith('v67') else 'Retain selected300:44/80 versus source39/80 on unchanged normalnear40medium40'
    record['next_action']='Superseded by explicit discrete-history prediction experiment v69'
    save(directory/'experiment_record.json',record)
record=dict(state='complete_rejected_at_training_holdout',goal_complete=False,source='v68selected300',
    source_navigation=dict(successes=44,episodes=80),user_permission='Neural next-state predictor may read historical discrete states',
    architecture='Frozen source perception, definitions, context, goals and base transitions. One GRUCell192 reads last8 discrete observations interpreted with fixed program definitions and historical actions. Residual definition-compatible decoder outputs next64-state probabilities for each candidate action.',
    neural_actor=False,predicts_actions=False,manual_semantic_states=False,maps=False,
    train_data=dict(episodes=160,transition_rows=13286),holdout_data=dict(episodes=40,transition_rows=3375),
    train_scope='normal near/medium',gpus=[6,7],updates=1000,
    planner_adapter='Root conditional next-state probabilities; continuation is existing discrete-table Bellman derivation. It does not yet propagate full history through every imagined branch.',
    planner_contract=dict(cases=24,max_probability_error=0,all_first_actions_identical_at_baseline=True),
    result=current,full80_evaluation_run=False,
    conclusion='Clear overfitting; no trained candidate passed predeclared heldout accuracy/NLL screen. Preserve v68 navigation model44/80.',
    next_action='Improve generalization of history-conditioned discrete dynamics before any further full80 navigation run; consider reducing episode-specific context memorization and collecting more normal near/medium measured transitions.')
save(folder/'experiment_record.json',record)
ledger=read(parent/'goal_progress.json')
ledger.update(authoritative_record=str(folder/'experiment_record.json'),remote_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_history_transition_v69',
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_history_transition_v69',
    phase='v69_complete_rejected_history_overfit',last_remote_check_unix=current['time'],
    next_regular_check_not_before_unix=None,supervisor_pid=None,rank_pids=[],latest_actual_update=1000,
    main_training_started=False,last_completed_training_update=1000,
    latest_verified_current_curriculum=dict(source='v68step300',successes=44,episodes=80,near_success=23,near_tasks=40,medium_success=21,medium_tasks=40),
    best_verified_current_curriculum=dict(source='v68step300',successes=44,episodes=80,sha256='dba453f4772d7a201ba9b595edd5952d24dd4221bb3b0f41c7d253a15c96f3ed'),
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    current_goal_turn_evidence='v68 fullnew80 improved39->44; explicit-history neural predictor v69 implemented and trained1000updates onGPU6/7; failed heldout screen so no navigation rerun. Actual supervisor/rank handles exited and GPUs6/7 free.',
    next_action=record['next_action'])
ledger['invariants']['neural_history_conditioned_next_state_prediction_allowed']=True
ledger['invariants']['neural_action_selection_allowed']=False
save(parent/'goal_progress.json',ledger)
print(json.dumps(dict(best_navigation=ledger['best_verified_current_curriculum'],history_experiment_state=record['state'],goal_complete=False)))
