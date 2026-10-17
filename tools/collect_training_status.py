import json,subprocess,time
from pathlib import Path

root=Path('/home/user/xjy/nesy_vla_20261010');run=root/'runs/full_v1'
def read(p):return json.loads(p.read_text()) if p.exists() else None
records=[]
for rank in range(6):
    p=run/f'rank{rank}.jsonl'
    if not p.exists():continue
    row=json.loads(p.read_text().splitlines()[-1])
    records.append({k:row.get(k) for k in ['rank','update','seconds','episodes','successes','gpu_peak_gib',
        'gpu_reserved_peak_gib','training_chunk_batch','next_training_chunk_batch','training_frames','phase_seconds','gradient_by_module']})
result={'time':time.time(),'supervisor':read(run/'supervisor.json'),'status':read(run/'status.json'),
    'committed':read(run/'latest/COMMITTED'),'ranks':records,'capacity_tuning':[read(run/f'batch_tuning_rank{r}.json') for r in range(6)],
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
    'free_disk':subprocess.check_output(['df','-h',str(root)],text=True),
    'latest_log_tail':(run/'launch_000.log').read_text(errors='replace')[-2400:]}
(root/'runs/current_status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
