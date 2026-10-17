import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
record=json.loads((root/'evidence/boundary_graph_fair_benchmark_v5.json').read_text())
old=sum(r['reactive_same_fast_encoder_seconds'] for r in record['replay'])
new=sum(r['predictive_batched_seconds'] for r in record['replay'])
evidence=dict(source='evidence/boundary_graph_fair_benchmark_v5.json',training_images=record['images'],
    changed_tokens=record['changed_tokens'],total_tokens=record['tokens'],
    identical_programs=record['identical_programs'],program_count=record['program_count'],
    shared_encoder_reactive_seconds=old,shared_encoder_predictive_seconds=new,replay_speedup=old/new,
    neural_calls_between_boundaries_rejected=True,
    scope='four fixed 21-image training replays; compute only, no physical execution',
    full_navigation_latency_proven=False,success_parity_proven=False,
    active_training_source_changed=False,standalone_evaluator='diagnostics/evaluate_visual_graph_fast.py')
progress=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'));progress['inference_speed_evidence']=evidence
progress['last_record_update_unix']=time.time()
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(evidence))
