import json,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010/symbolic_ppo8_v82')
rows=[]
for p in sorted(B.glob('rank*/processes.json')):
    j=json.loads(p.read_text());rss=[]
    for pid in j['simulator_pids']:
        status=Path(f'/proc/{pid}/status')
        if status.exists():
            d=dict(x.split(':',1) for x in status.read_text().splitlines() if ':' in x)
            rss.append(int(d.get('VmRSS','0 kB').split()[0])/1024)
    rows.append({'rank':p.parent.name,'simulator_count':len(rss),'total_sim_rss_mib':sum(rss),'max_sim_rss_mib':max(rss,default=0)})
memory={k:v.strip() for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
print(json.dumps({'time':time.time(),'memory':{k:memory[k] for k in ['MemTotal','MemAvailable','SwapFree']},'simulators':rows},indent=2))
