"""Record verified terminal evidence and the newly launched collection."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
result=read(BASE/'relational_rules_v62/result.json')
observation=read(BASE/'relational_rules_v62/last_observation.json')
assert observation['status']['state']=='complete' and not observation['command']
last=result['history'][-1];assert last['update']==1200
control=last['results']['holdout']['control'];candidate=last['results']['holdout']['relational']
passed=(candidate['accuracy']>control['accuracy']+.02 and candidate['changed_accuracy']>control['changed_accuracy']+.02 and candidate['nll']<control['nll'])
assert not passed
record=dict(state='complete_not_promoted',authoritative_observation=str(BASE/'relational_rules_v62/last_observation.json'),
    fixed_final_update=1200,control=control,candidate=candidate,screen_passed=False,
    conclusion='More fitting capacity increased fit accuracy but failed the preselected held-out transition test. Do not run a navigation trial or replace v60.',
    navigation_evaluations=0,finalfixed80_used=False)
(BASE/'relational_rules_v62/experiment_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
live=read(HERE/'last_observation.json')
assert live['status']['state']=='collecting' and len([p for p in live['processes'] if p['command']])==3
ledger=read(BASE/'goal_progress.json')
ledger.update(goal_status='active',completion_proven=False,phase='collecting_measured_two_step_successors',
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    previous_goal_turn_evidence='v62 fixed-label relational pilot completed and failed its predetermined held-out screen; v63 collector code prepared.',
    main_training_started=False,active_evaluation_coordinator_pid=None,active_evaluation_worker_pids=[],
    supervisor_pid=None,rank_pids=[],collection_coordinator_pid=live['launch']['pid'],
    collection_rank_pids=live['status']['workers'],last_collection_check_unix=live['time'],
    next_collection_check_not_before_unix=live['time']+630,
    authoritative_record=str(HERE/'experiment_record.json'),
    next_action='Verify initial physical two-step probe checks and actual completed episode files. Then collect all600 original training tasks; train categorical1/2-step rules with frozen source abstraction. Never use finalfixed80 for model selection.')
exp=dict(state='collecting',source=live['launch']['source_commit'],remote_observation=live,
    goal_success_not_proven=True,best_navigation=ledger['best_verified_full80'],
    training_tasks=600,training_source_only=True,future_targets='Frozen encoder discrete IDs from real simulator outcomes; no semantic or spatial labels',
    actual_actions='Unchanged pure NumPy categorical proof planner',
    planned_training='One-step CE, predicted-memory/observed-observation two-step CE, and all64-intermediate-observation marginal likelihood',
    reference='https://planetrl.github.io/',
    adaptation='Uses multi-step latent prediction supervision as motivation. Not PlaNet: no continuous recurrent planning state, CEM, reward model, or per-action memory filtering.',
    evaluation='Original dev80 unchanged; independent finalfixed80 untouched')
(HERE/'experiment_record.json').write_text(json.dumps(exp,indent=2),encoding='utf-8')
(BASE/'goal_progress.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
print(json.dumps(dict(v62='not_promoted',v63='collecting',pids=live['status']['workers'],next_check=live['time']+630)))
