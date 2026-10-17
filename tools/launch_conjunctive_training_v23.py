import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_conjunctive_v23'
assert not (root/'processes.json').exists(),'Inspect the existing processes instead of relaunching'
probe=json.loads((base/'conjunctive_probe_v22/completion_inspection.json').read_text())
assert probe['results']['conjunctive_temperature_4']['state']=='complete'
assert probe['results']['conjunctive_temperature_4']['last']['step']==800
finished=json.loads((base/'symbolic_task_v15/completion_inspection.json').read_text())
assert not finished['pending'] and finished['status']['state']=='complete'
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(a):int(b) for a,b in (line.split(',') for line in memory.splitlines())}
assert all(free[g]>21000 for g in range(6)),'Do not disturb other GPU processes'
assert shutil.disk_usage(base).free>5*2**30
records=[]
for name,gpus in [('branch_control','0,1,2'),('conjunctive','3,4,5')]:
    out=root/'runs'/name;out.mkdir(parents=True)
    command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','scripts/supervise.py',
        '--config',str(root/'configs'/f'{name}.json'),'--out',str(out),'--gpus',gpus]
    with (out/'supervisor.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    record={'name':name,'supervisor_pid':child.pid,'time':time.time(),'command':command,'gpus':gpus}
    records.append(record);(root/'processes.json').write_text(json.dumps(records,indent=2))
experiment=json.loads((root/'experiment.json').read_text())
experiment.update(state='launched',selection='Temperature4 avoids the worst probe gradient explosion; matched conjunction-vs-branch full-trajectory study.',
    capacity_evidence='Conjunction train47/72 vs branch44/72; heldout2/26 vs3/26. This is not evidence of navigation or generalization improvement.',
    counterfactual_gradient_enabled=False)
(root/'experiment.json').write_text(json.dumps(experiment,indent=2));print(json.dumps(records,indent=2))
