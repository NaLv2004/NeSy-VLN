import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';study=base/'conjunctive_probe_v22'
probe=json.loads((base/'counterfactual_branch_v20/completion_inspection.json').read_text())
assert all(v['state']=='complete' for v in probe['results'].values())
unused=base/'symbolic_counterfactual_v21/experiment.json';record=json.loads(unused.read_text())
record.update(state='not_started_negative_capacity_probe',reason='Counterfactual estimator did not improve matched first-action capacity or held-out predictions. No long training launched.')
unused.write_text(json.dumps(record,indent=2))
assert not study.exists();study.mkdir();records=[]
for temperature in [1.,4.]:
    name='conjunctive_temperature_'+str(int(temperature));out=study/name;out.mkdir()
    command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(base/'probe_paired_goal_learning.py'),
        '--root',str(root),'--data',str(base/'paired_goal_probe_v12/input_snapshot'),
        '--checkpoint',str(base/'symbolic_reward_v11/initialization_source'),'--out',str(out),
        '--steps','800','--task-token-encoder','--conjunctive-rules','--route-temperature',str(temperature)]
    with (out/'probe.log').open('ab') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    record={'name':name,'pid':p.pid,'command':command,'time':time.time()};records.append(record)
    (out/'process.json').write_text(json.dumps(record,indent=2))
(study/'processes.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
