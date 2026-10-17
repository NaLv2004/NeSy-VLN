"""Freeze a committed model before a read-only training-transition diagnosis."""
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'diagnosis_anonymous_v5';out.mkdir(exist_ok=True)
if (out/'launcher.json').exists():raise RuntimeError('Inspect existing diagnostic process; do not duplicate')
source=root/'run_anonymous_v5/best_development.pt'
for _ in range(3):
    before=json.loads(source.with_suffix('.COMMITTED.json').read_text())
    data=source.read_bytes()
    after=json.loads(source.with_suffix('.COMMITTED.json').read_text())
    if before==after and hashlib.sha256(data).hexdigest()==before['sha256']:break
else:raise RuntimeError('Checkpoint publication in progress; retry later without disturbing training')
(out/'frozen.pt').write_bytes(data);(out/'frozen.COMMITTED.json').write_text(json.dumps(before,indent=2))
used=int(subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>200:raise RuntimeError('GPU4 has another process; leave it alone')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(root/'src_anonymous_v5'))
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(root/'diagnose_anonymous_training.py'),
    '--checkpoint',str(out/'frozen.pt'),'--data',str(root/'counterfactual_own_policy_v1'),'--out',str(out/'summary.json'),
    '--batches','64','--device','cuda']
log=(out/'driver.log').open('ab',buffering=0)
p=subprocess.Popen(cmd,cwd=root/'src_anonymous_v5',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,gpu=4,checkpoint=before,time=time.time(),command=cmd)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
