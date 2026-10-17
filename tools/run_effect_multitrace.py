import json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/discrete_v4')
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
records=[]
for update in [1,10,20]:
    folder=root/f'runs/discrete_gru/demonstration_traces/update_{update:06d}'
    destination=root/f'diagnostics/effect_probe_{update:06d}.json'
    if not (folder/'trajectory.json').exists():continue
    if not destination.exists():
        with destination.with_suffix('.log').open('w') as log:
            subprocess.run([python,str(root/'scripts/probe_effect_learning.py'),'--folder',str(folder),
                '--output',str(destination)],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads(destination.read_text())
    records.append({'update':update,'frames':result['frames'],'changed_bits':result['changed_bits'],
        'copy_accuracy':result['copy_accuracy'],'trials':[{k:v for k,v in trial.items() if k!='history'}|
        {'final':trial['history'][-1]} for trial in result['trials']]})
out=root/'diagnostics/effect_multitrace.json'
out.write_text(json.dumps({'time':time.time(),'results':records},indent=2))
print(out.read_text())
