import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'anonymous_information_trial';src=root/'src_anonymous_information_trial'
previous=root/'run_rollout_proof_v8r1'
status=json.loads((previous/'status.json').read_text());launcher=json.loads((previous/'launcher.json').read_text())
assert status['state']=='training_budget_complete' and not (Path('/proc')/str(launcher['pid'])).exists()
if (out/'parent_launcher.json').exists() or (out/'launcher.json').exists():raise RuntimeError('Trial already launched')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu.splitlines()}
if memory[7]>300:raise RuntimeError('GPU7 busy; no process stopped')
out.mkdir(exist_ok=True);log=(out/'driver.log').open('ab',buffering=0)
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(src/'trial_anonymous_information.py')]
p=subprocess.Popen(cmd,cwd=src,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),source=str(src),command=cmd,updates=600,gpu=7)
(out/'parent_launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
