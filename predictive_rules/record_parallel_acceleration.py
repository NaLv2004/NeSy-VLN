import json,time
from pathlib import Path
BASE=Path(__file__).parent;ledger_path=BASE/'goal_progress.json';ledger=json.loads(ledger_path.read_text())
launch=json.loads((BASE/'parallel_v75_v76_launch.json').read_text())
benchmark=json.loads((BASE/'gpu_efficiency_v74/benchmark.json').read_text())
results={name:json.loads((BASE/name/'result.json').read_text()) for name in ('continued_evidence_v72','compact_history_v73')}
record=dict(state='parallel_training_and_collection',goal_complete=False,user_authorized_gpus=[2,3,4,5,6,7],
    launches=launch,completed_candidates=results,acceleration=benchmark,
    best_navigation=dict(source='v68selected300',successes=44,episodes=80,retained=True),
    collection=dict(run='diverse_history_v70',completed_at_last_check=816,total=1200,last_check=1792214741.346717),
    waiting_full_data_training=dict(run='run_expanded_dynamics_v71',pid=2264606,gpus=[6,7],
        gpu_symbolic_planning=True,measured_autobatch=True,validation_workers_per_gpu=40),
    changes=dict(v75='Frozen44source perception and memory; train only next-O rules on768fresh episodes',
        v76='Compacttwo-observation predictor on same768episodes; larger throughput-tuned batches',
        gpu_planning='Identical float64 table recurrence on GPU; CPU proof construction/audit retained;8realprogram524288decisionsallidentical'),
    next='Inspect v75/v76 exact completion events; adopt only genuine80task improvement, not prediction metric. Check collection again only after1792215372. v71 starts after actual collection exit. Preserve source44 and all fixed splits.')
(BASE/'gpu_efficiency_v74/experiment_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
ledger['invariants']['gpus']=[2,3,4,5,6,7]
ledger.update(authoritative_record=str(BASE/'gpu_efficiency_v74/experiment_record.json'),
    phase='v75_v76_large_batch_gpu_symbolic_parallel_with_v70_collection',
    active_parallel_runs=launch['launched'],main_training_started=True,
    last_collection_check_unix=1792214741.346717,next_collection_check_not_before_unix=1792215372,
    last_remote_check_unix=1792214958.7301004,next_regular_check_not_before_unix=1792215589,
    latest_actual_update=143,current_goal_turn_classification='progress',
    current_goal_turn_evidence='Completedv72=35/80,v73=43/80;44sourcepreserved. GPUfloat64Bellman8programstest27.88xfaster with524288identicaldecisions. Same-modeltrainingbatch1024to6144 doublesrowthroughput at20.04GiB. Launchedv75/v76 on2-5 with768preselectedcommittedepisodes609fit159heldout,actualv75update143batch8192peak16.40GiB,v76completed1000updatesselected500andenteredGPU80evaluation. v70on6/7last816/1200. Pendingv71updatedtoGPUplannerandautobatch.',
    next_action=record['next'])
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2))
print(json.dumps(dict(record=str(BASE/'gpu_efficiency_v74/experiment_record.json'),goal_complete=False)))
