import json, subprocess, time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010/original_ppo_resume_v99')
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

records=[json.loads(l) for l in (B/'rank0/updates.jsonl').read_text().splitlines() if l.strip()]
records=[x for x in records if x['update']<=r['checkpoint_preserved']['epoch']]
stats={}
for name,w in [('all',records),('last100',records[-100:]),('last10',records[-10:])]:
 n=sum(x['training']['episodes'] for x in w);k=sum(x['training']['successes'] for x in w)
 stats[name]=dict(first_update=w[0]['update'],last_update=w[-1]['update'],updates=len(w),episodes=n,successes=k,success_rate=k/n,mean_update_seconds=sum(x['seconds'] for x in w)/len(w))
report=dict(time=time.time(),stopped=r,training=stats,evaluations={p.parent.name:json.loads(p.read_text()) for p in sorted((B/'evaluations').glob('*/summary.json'))})
(B/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2))
print('TRAINING_SUMMARY',json.dumps(stats))
