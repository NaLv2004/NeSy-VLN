import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_recovery_v28'
assert not (root/'processes.json').exists(),'Inspect the existing processes instead of relaunching'
completion=json.loads((base/'symbolic_conjunctive_v23/completion_inspection.json').read_text())
assert not completion['pending'] and all(j['status']['state']=='complete' and not j['live'] for j in completion['jobs'].values())
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(a):int(b) for a,b in (line.split(',') for line in memory.splitlines())}
assert all(free[g]>21000 for g in range(6)),'Do not disturb other GPU processes'
assert shutil.disk_usage(base).free>5*2**30
records=[]
for name,gpus in [('base_control','0,1,2'),('recovery_mixture','3,4,5')]:
    out=root/'runs'/name;out.mkdir(parents=True)
    command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','scripts/supervise.py',
        '--config',str(root/'configs'/f'{name}.json'),'--out',str(out),'--gpus',gpus]
    with (out/'supervisor.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    record={'name':name,'supervisor_pid':child.pid,'time':time.time(),'command':command,'gpus':gpus}
    records.append(record);(root/'processes.json').write_text(json.dumps(records,indent=2))
experiment=json.loads((root/'experiment.json').read_text())
experiment.update(state='launched',treatment='25% complete successful recovery demonstrations, with random prefixes masked from action imitation.')
(root/'experiment.json').write_text(json.dumps(experiment,indent=2));print(json.dumps(records,indent=2))
