"""Launch the reviewed full-trajectory outcome comparison on six free GPUs."""
import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_execution_v38';read=lambda p:json.loads(p.read_text())
assert not (root/'processes.json').exists(),'Inspect existing processes instead of relaunching'
probe=read(base/'neural_conjunction_learning_v37/result.json');assert probe['state']=='complete'
pid=read(base/'neural_conjunction_learning_v37/launcher.json')['pid'];proc=Path(f'/proc/{pid}/cmdline')
assert not proc.exists() or b'probe_neural_conjunctions_v37.py' not in proc.read_bytes(),'Probe still owns GPU6'
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(a):int(b) for a,b in (line.split(',') for line in memory.splitlines())}
assert all(free[g]>21000 for g in [1,2,3,4,5,6]),'Do not disturb another process'
assert shutil.disk_usage(base).free>5*2**30,'Reserve room for resumable checkpoints'
records=[]
for name,gpus in [('detached_outcome_control','1,2,3'),('grounded_outcome_feedback','4,5,6')]:
    out=root/'runs'/name;out.mkdir(parents=True)
    command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','scripts/supervise.py',
        '--config',str(root/'configs'/f'{name}.json'),'--out',str(out),'--gpus',gpus]
    with (out/'supervisor.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    records.append({'name':name,'supervisor_pid':child.pid,'time':time.time(),'command':command,'gpus':gpus})
    (root/'processes.json').write_text(json.dumps(records,indent=2))
experiment=read(root/'experiment.json');experiment.update(state='launched',probe_review=read(root/'probe_review.json'))
(root/'experiment.json').write_text(json.dumps(experiment,indent=2));print(json.dumps(records,indent=2))
