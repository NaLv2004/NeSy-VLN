import json
from pathlib import Path
root=Path(__file__).resolve().parent
path=root/'progress.json';p=json.loads(path.read_text(encoding='utf-8'))
m=json.loads((root/'monitor_latest.json').read_text(encoding='utf-8'))
launch=json.loads((root/'parallel_render_launcher.json').read_text(encoding='utf-8'))
p['previous_goal_turn_classification']='progress: raw grounding jointly implemented and training verified at update127'
p['this_turn_classification']='progress: parallelized training-data rendering, resumed all completed data, verified live train1500 and two scheduled development results'
p['runtime'].update(latest_verified_update=m['status']['update'],last_update_seconds=m['status']['seconds'],
    committed_update=m['latest_commit']['update'],committed_sha256=m['latest_commit']['sha256'],
    observed_at_remote_epoch=m['observed_at'],next_routine_check_not_before_remote_epoch=m['observed_at']+600,
    process_alive_verified=m['processes'].get(str(p['runtime']['pid'])) is not None)
p['collection'].update(pid=launch['pid'],superseded_pid=launch['supersedes'],rendering_workers=4,
    parallel_launcher='raw_successor_features_v31/parallel_render_launcher.json',
    status_path='raw_successor_features_v31/parallel_render/status.json',positions_available=m['raw_positions'],
    episodes_completed=m['collector_status']['completed'],process_alive_verified=m['processes'].get(str(launch['pid'])) is not None,
    note='Only GPU6 used. Four independent simulator workers overlap rendering with one serialized neural encoder. Same recorded train actions, provenance and completed examples preserved.')
p['scheduled_development_results']=m['validation']
p['training_window_metrics']=m['training_windows']
p['current_conclusion']='v31step1000 reached17/80, below preserved v29best19/80. It satisfies>=5actions/program but not>=48/80. No benefit claim and no extra diagnostic evaluations.'
p['next_action']=f"No routine live check before {m['observed_at']+600}. Then read exact trainingPID1840654, collectorPID{launch['pid']}, scheduled results and COMMITTED checkpoint. Preserve unrelated GPU0..5 work. Keep training while full raw branch data arrives; do not start diagnostic comparisons or change the fixed development tasks. Repair concrete errors if present."
path.write_text(json.dumps(p,indent=2),encoding='utf-8')
print('Progress saved; goal remains active.')
