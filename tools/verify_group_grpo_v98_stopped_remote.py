import json, subprocess, time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010/group_grpo_v98')
r=json.loads((B/'USER_STOP.json').read_text())
remaining=[]
for old in r['processes']:
    p=Path('/proc')/str(old['pid'])
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[19]==old['start'] and stat[0]!='Z':
            remaining.append(dict(pid=old['pid'],state=stat[0],wchan=(p/'wchan').read_text()))
    except FileNotFoundError:
        pass
r['remaining']=remaining
r['verified_at']=time.time()
r['gpus_after']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
r['gpu_processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
r['state']='stopped' if not remaining else 'stopped_waiting_for_kernel_cleanup'
(B/'USER_STOP.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k not in ['processes','latest_validation']},indent=2))
