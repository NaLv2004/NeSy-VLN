import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'semantic_component_v69';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists()
corpus=b/'semantic_tokens_v68/corpus'
for rank in range(8):
 r=json.loads((corpus/f'rank{rank}.result.json').read_text());assert r['completed']==r['total']
for gpu in [0,1,2]:
 value=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 assert int(value.strip())<1000
source=b/'symbolic_paired_v12'
shutil.copytree(source/'nesy_vla',out/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(source/'scripts/probe_semantic_tokens_v69.py',out/'probe_semantic_tokens_v69.py')
jobs=[]
for gpu,arm in enumerate(['visual_only','aligned_text','shuffled_text']):
 folder=out/arm;folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'probe_semantic_tokens_v69.py'),'--arm',arm]
 with (folder/'driver.log').open('ab') as log:
  child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 jobs.append({'pid':child.pid,'gpu':gpu,'arm':arm,'command':command})
r={'time':time.time(),'jobs':jobs,'trainer_sha256':hashlib.sha256((out/'probe_semantic_tokens_v69.py').read_bytes()).hexdigest(),
 'semantic_module_sha256':hashlib.sha256((out/'nesy_vla/semantic_token_automaton.py').read_bytes()).hexdigest()}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
