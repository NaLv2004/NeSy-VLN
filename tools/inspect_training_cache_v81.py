import json,os,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010')
roots=['/home/user/xjy/vamos_4x4090_20260906','/home/user/xjy/objectnav_qwen35_rl_20260921','/home/user/xjy/evaluations','/home/user/xjy/repro']
result={}
for root in roots:
 p=subprocess.run(['du','-x','-B1','--max-depth=2',root],capture_output=True,text=True,timeout=100)
 rows=[]
 for line in p.stdout.splitlines():
  f=line.split('\t',1)
  if len(f)==2:rows.append((int(f[0]),f[1]))
 result[root]=[{'gib':round(n/2**30,3),'path':p} for n,p in sorted(rows,reverse=True)[:20]]
cfg=json.loads(Path('/home/user/xjy/lite3_photo_world_20260927/lab_rl64/run32_v4_photo25/config.json').read_text())
result['source_qwen_reward']={k:v for k,v in cfg.items() if any(w in k for w in ['reward','penalty','collision'])}
stopped=b/'perceptual_rl_v80/action_rules_only';pids=json.loads((stopped/'processes.json').read_text())['simulator_pids']
result['stopped_control_handles']={str(pid):Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{pid}/cmdline').exists() else None for pid in pids}
(b/'training_cache_inspection_v81.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
