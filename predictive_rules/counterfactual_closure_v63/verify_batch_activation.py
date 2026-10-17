import json,time,subprocess
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
DATA=ROOT/'counterfactual_closure_v63'
read=lambda p:json.loads(p.read_text())
launch=read(DATA/'launcher.json');ranks=[read(DATA/f'status_rank{i}.json') for i in (0,1)]
examples=[]
for p in sorted((DATA/'episodes').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:60]:
    r=read(p)
    if r.get('successor_encoding')=='batched_51_bfloat16':
        examples.append(dict(episode=r['episode'],positions=r['positions'],encoding=r['successor_encoding']))
        if len(examples)==2:break
result=dict(time=time.time(),ranks=ranks,batched_committed_examples=examples,
    completed=sum(r['completed'] for r in ranks),gpu=subprocess.check_output(
    ['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
    owned_alive=[p for p in [launch['pid'],*[r['pid'] for r in ranks]] if Path(f'/proc/{p}/cmdline').exists()],
    memory_info=[s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemTotal:','MemAvailable:'))])
(DATA/'batch_activation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
