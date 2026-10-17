import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'subgoal_rule_inverse_trial';src=root/'src_subgoal_v2'
out.mkdir(exist_ok=True)
if (out/'launcher.json').exists():raise RuntimeError('Trial already launched')
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in raw.splitlines()}
if memory[4]>300:raise RuntimeError('GPU4 is busy; leaving its processes intact')
script='''import subprocess,sys
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
subprocess.run([sys.executable,'-u','pilot_subgoal.py'],check=True)
subprocess.run([sys.executable,'-u','diagnose_subgoal_training.py','--checkpoint',str(r/'subgoal_rule_inverse_trial/candidate.pt'),'--out',str(r/'subgoal_rule_inverse_trial/training_diagnosis.json')],check=True)
subprocess.run([sys.executable,'-u','evaluate_subgoal.py','--checkpoint',str(r/'subgoal_rule_inverse_trial/candidate.pt'),'--out',str(r/'subgoal_rule_inverse_trial/development')],check=True)
'''
driver=out/'run.py';driver.write_text(script)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',PYTHONPATH=str(src),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
log=(out/'driver.log').open('ab',buffering=0)
p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(driver)],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=p.pid,time=time.time(),gpu=4,source=str(src),updates=400)
(out/'launcher.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
