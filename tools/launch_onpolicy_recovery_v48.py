import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'data/onpolicy_recovery_v48';assert (out/'manifest.json').exists() and not (out/'launcher.json').exists()
assert shutil.disk_usage(b).free>3*2**30
for gpu in range(1,7):
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>16000,(gpu,free)
src=out/'src';src.mkdir(exist_ok=True);shutil.copy2(b/'collect_onpolicy_recovery_v48.py',src/'collector.py');jobs=[]
for rank,gpu in enumerate(range(1,7)):
 run=out/f'rank_{rank}';run.mkdir(exist_ok=True)
 with (run/'driver.log').open('ab') as log:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  proc=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/sim/bin/python','-u',str(src/'collector.py'),'--rank',str(rank),'--world-size','6'],env=env,cwd=src,stdout=log,stderr=log,start_new_session=True)
 jobs.append({'rank':rank,'gpu':gpu,'pid':proc.pid,'time':time.time(),'script':str(src/'collector.py'),'script_sha256':hashlib.sha256((src/'collector.py').read_bytes()).hexdigest()})
(out/'launcher.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs,indent=2))
