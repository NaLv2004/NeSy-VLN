"""Bounded, read-only progress snapshot. No restarting on observation failure."""
import json
import subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_v1')
for name in ['launcher.json','status.json','latest.COMMITTED.json','best_development.json']:
    path=root/name
    if path.exists():
        print(name,path.read_text())
if (root/'launcher.json').exists():
    pid=json.loads((root/'launcher.json').read_text())['pid']
    print('PROCESS',subprocess.getoutput(f'ps -p {pid} -o pid,ppid,stat,etime,pcpu,pmem,args'))
for suite in ['development','fixed80']:
    path=root/'baseline_1097'/suite/'summary.json'
    if path.exists():print('BASELINE',suite,path.read_text())
for name in ['train.jsonl','validation.jsonl','driver.log']:
    path=root/name
    if path.exists():print(name,'\n'.join(path.read_text(errors='replace').splitlines()[-6:]))
print('GPU',subprocess.getoutput('nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader'))
print('GPU_PROCESSES',subprocess.getoutput('nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader'))
