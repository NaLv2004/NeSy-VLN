"""Wait for actual collector handles, audit data, then run two isolated rule trials."""
import hashlib,json,os,selectors,shutil,subprocess,sys,time,traceback
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');data=b/'data/onpolicy_recovery_v48';out=b/'recovery_rule_learning_v49';out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());tmp=out/'pipeline_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'pipeline_status.json');print(json.dumps(state),flush=True)
try:
 selector=selectors.DefaultSelector();watched=[]
 for job in read(data/'launcher.json'):
  pid=job['pid'];p=Path(f'/proc/{pid}/cmdline')
  if not p.exists():continue
  command=p.read_bytes();assert str(data/'src/collector.py').encode() in command and ('--rank '+str(job['rank'])).encode() in command.replace(b'\0',b' ')
  try:fd=os.pidfd_open(pid)
  except ProcessLookupError:continue
  selector.register(fd,selectors.EVENT_READ);watched.append(pid)
 progress(state='waiting_for_collectors',verified_live_pids=watched)
 while selector.get_map():
  for key,event in selector.select():selector.unregister(key.fd);os.close(key.fd)
 selector.close()
 for rank in range(6):
  assert read(data/f'rank_{rank}/status.json')['state']=='complete',('Collector incomplete',rank)
 policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
 progress(state='auditing_completed_dataset')
 subprocess.run([policy,str(out/'src/merge.py')],check=True)
 progress(state='encoding_corrective_training_images',dataset=read(data/'COMPLETE'))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 subprocess.run([policy,str(out/'src/encode.py')],env=env,check=True)
 assert shutil.disk_usage(b).free>2.5*2**30
 children=[];jobs=[]
 for arm,gpu in [('with_action',1),('without_previous_action',2)]:
  free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>16000,(gpu,free)
  folder=out/arm;folder.mkdir(exist_ok=True);assert not (folder/'launcher.json').exists()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(b/'induced_boolean_rules_v41_deps'))
  with (folder/'driver.log').open('ab') as log:
   p=subprocess.Popen([policy,'-u',str(out/'src/probe.py'),'--arm',arm],cwd=out/'src',env=env,stdout=log,stderr=log)
  row={'pid':p.pid,'gpu':gpu,'arm':arm,'time':time.time(),'script':str(out/'src/probe.py'),'script_sha256':hashlib.sha256((out/'src/probe.py').read_bytes()).hexdigest(),'data_split_unchanged':True,'original_fixed80_used':False};(folder/'launcher.json').write_text(json.dumps(row,indent=2));jobs.append(row);children.append(p)
 progress(state='training_and_evaluating',jobs=jobs)
 codes=[p.wait() for p in children];assert codes==[0,0],codes
 summaries={j['arm']:read(out/j['arm']/'development/summary.json') for j in jobs};progress(state='complete',summaries=summaries)
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
