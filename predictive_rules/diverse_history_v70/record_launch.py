import json
from pathlib import Path
folder=Path(__file__).parent;parent=folder.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,row):p.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
launch=read(folder/'launch.json');first=read(folder/'first_data_observation.json');next_run=read(parent/'expanded_dynamics_v71/training_launch.json')
record=dict(state='collecting_training_data',goal_complete=False,source_navigation_successes=44,source_navigation_episodes=80,
    launch=launch,first_committed_evidence=first,tasks=1200,fit_episodes=960,holdout_episodes=240,
    previous200excluded=True,validation_layouts_excluded=True,normal_only=True,near600_medium600=True,
    current80_unchanged=True,finalfixed80_used=False,all_executed_actions_symbolically_derived=True,
    sampling='Frozen v68 policy actual trajectories; all17firstactions and one continuingsecond for eachnonSTOP first every4thdecision and atSTOP/failure. Alternative images only training labels.',
    research=[dict(title='Predictive Representations of State',url='https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html',
        relevance='Motivation for action-conditional future-discriminating state; collection is our adaptation, not a reproduction')],
    next_training_handoff=next_run,
    next_check_not_before_unix=first['time']+630)
save(folder/'experiment_record.json',record)
next_record=dict(state='waiting_for_actual_collection_exit',goal_complete=False,launch=next_run,
    source='v68selected300,44/80 on unchanged normalnear40medium40',
    optimized='Generic visual-to-definition residual and next-observation rule decoder only',
    frozen='Visual backbone, original observation/definition generators, context generator, goal generator, memory definitions, memory predictor, action embeddings',
    losses='Joint visual goal-code evidence and one/two-step categorical dynamics; fixed-sequence symbolic goal probability calibration from existing rule rows',
    memory_alignment='Use recorded deployed M at each actual action rather than silently re-grounding M from actual observation every step. Grounding fallback only at failed rule-renewal positions without executed records.',
    dataset='1200new normalnear/medium train-layout episodes,960fit240heldout; no development optimization',
    selection='Training-layout holdout screen before one full unchanged80trial; no repetitive source44/80 reevaluation',
    training_has_not_started=True,gpus=[6,7],preserve_other_gpu_jobs=True)
save(parent/'expanded_dynamics_v71/experiment_record.json',next_record)
ledger=read(parent/'goal_progress.json')
ledger.update(authoritative_record=str(folder/'experiment_record.json'),phase='v70_collection_with_v71_training_handoff',
    remote_run=launch['run'],remote_source=launch['source'],remote_future_training_run=next_run['run'],
    last_remote_check_unix=first['time'],next_regular_check_not_before_unix=first['time']+630,
    collection_coordinator_pid=launch['pid'],collection_rank_pids=first['collection_state']['workers'],
    last_collection_check_unix=first['time'],next_collection_check_not_before_unix=first['time']+630,
    training_handoff_pid=next_run['pid'],supervisor_pid=next_run['pid'],rank_pids=[],latest_actual_update=0,
    main_training_started=False,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
    current_goal_turn_evidence='Verified v69 terminal handles absent. Measured identical-premise future ambiguity503groups versus10 with one historical observation on fit, with reduced repeated coverage. Launched1200fresh normalnear/medium collection;96committedepisodes observed. Exact-process-exit v71handoff verified live; current training uses recorded rollout M.',
    next_action='Wait at least630seconds from last collection check. Inspect exact coordinator2259919, collectors2259920/2259921, training handoff2264606. If collector terminal verify complete1200 and actual training child before any restart. v71 source complete; if failure investigate actual state and preserve committed data. Keep44/80 until measured improvement; objective>80 remains unachieved.')
save(parent/'goal_progress.json',ledger)
print(json.dumps(dict(collection_pid=launch['pid'],committed=first['committed_files'],training_handoff=next_run['pid'],next_regular_check=ledger['next_regular_check_not_before_unix'],goal_complete=False)))
