"""A bounded CPU posthoc diagnosis; no navigation restart and no GPU use."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'diagnosis_anonymous_own_v5';out.mkdir(exist_ok=True)
if (out/'launcher.json').exists():raise RuntimeError('Already launched; inspect its process')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(root/'src_anonymous_v6'))
command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(root/'diagnose_anonymous_own_rollouts.py'),
    '--checkpoint',str(root/'diagnosis_anonymous_v5/frozen.pt'),'--data',str(root/'counterfactual_anonymous_v5'),
    '--out',str(out),'--device','cpu']
log=(out/'driver.log').open('ab',buffering=0)
p=subprocess.Popen(command,cwd=root/'src_anonymous_v6',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),command=command)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
