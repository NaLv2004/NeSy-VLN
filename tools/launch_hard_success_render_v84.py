import json,os,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'hard_success_v84';out.mkdir(exist_ok=True)
if (out/'process.json').exists():
 old=json.loads((out/'process.json').read_text());proc=Path(f'/proc/{old["pid"]}/cmdline')
 assert not proc.exists() or not proc.read_bytes(),'Existing render process is still live'
 assert not (out/'status.json').exists() or json.loads((out/'status.json').read_text())['state']!='complete'
 (out/'process.json').rename(out/f'process_{old["pid"]}_terminal.json')
steps=[p for p in (B/'parallel_ppo_v84/rollouts').glob('step_*') if int(p.name.split('_')[-1])>=36 and len(list(p.glob('rank*/*.json')))==192]
if not steps:
 print(json.dumps({'ready':False,'reason':'Waiting for first completed enlarged-parallel training trajectories.'}));raise SystemExit
mem={k:v.strip() for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
assert int(mem['MemAvailable'].split()[0])>4*1024**2,'Leave sufficient system RAM for training'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/sim/bin/python','-u',str(B/'render_ppo_hard_successes_v84.py')],cwd=B,env=env,stdout=log,stderr=log,start_new_session=True)
result={'pid':p.pid,'time':time.time(),'command':str(B/'render_ppo_hard_successes_v84.py'),'sim_gpu':0,'new_policy_inference':False}
(out/'process.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
