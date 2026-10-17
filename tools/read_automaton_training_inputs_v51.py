import json,shutil,subprocess
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010')
print(json.dumps({'source_rows':json.loads((b/'spatial_automaton_v43/training_rows.json').read_text())[:2],
 'new_rows':json.loads((b/'data/onpolicy_recovery_v48/training_rows.json').read_text())[:2],
 'free_gib':shutil.disk_usage(b).free/2**30}))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
