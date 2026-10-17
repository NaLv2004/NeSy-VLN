import json,shutil,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010')
r={'free_gib':shutil.disk_usage(b).free/2**30,'files':[]}
for name in ['symbolic_execution_v38','symbolic_recovery_v28','goal_conditioned_automaton_v55','factorized_automaton_v51','critic_isolation_rl_v50']:
 for m in (b/name).rglob('COMMITTED'):
  c=json.loads(m.read_text());p=m.parent/c.get('checkpoint_file','checkpoint.pt')
  r['files'].append({'folder':str(m.parent.relative_to(b)),'size_mb':round(p.stat().st_size/2**20,1) if p.exists() else None,'links':p.stat().st_nlink if p.exists() else None,'optimizer':c.get('optimizer_included'),'epoch':c.get('epoch')})
print(json.dumps(r))
