"""Choose from completed development runs, verify replay, evaluate fixed80 once."""
import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'fixed_perceptual_v77';out.mkdir(exist_ok=True)
lock=(out/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
read=lambda p:json.loads(p.read_text())
assert not (out/'launcher.json').exists(),'Inspect existing evaluation; do not launch twice'
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
sources=[B/'perceptual_fsm_v74'/x for x in ('frozen','tail_1e4')]
sources += [B/'observation_inverse_v76'/x for x in ('observed_inverse','no_inverse')]
candidates=[]
for run in sources:
 supervisor=read(run/'supervisor.json')
 assert supervisor['state']=='complete' and supervisor['exit_code']==0,('Training not terminal',str(run))
 for pid in [supervisor['pid'],supervisor['supervisor_pid']]:
  assert not Path(f'/proc/{pid}/cmdline').exists(),('Old process still exists',pid)
 assert read(run/'status.json')['state']=='complete'
 for path in sorted((run/'evaluations').glob('*/summary.json')):
  result=read(path);assert result['episodes']==80
  checkpoint=run/path.parent.name;commit=read(checkpoint/'COMMITTED')
  candidates.append({'run':str(run),'checkpoint':str(checkpoint),'commit':commit,'development':result,'replay':str(path.parent)})
chosen=sorted(candidates,key=lambda x:(-x['development']['successes'],x['development']['collision_action_rate'],x['commit']['epoch'],x['run']))[0]
selection={'time':time.time(),'selection_uses':'Completed development success only, collision rate as tie-break, then earlier epoch',
 'candidates':candidates,'chosen':chosen,'fixed80_seen_for_selection':False,'horizon':32,'revision_interval':10}
if (out/'selection.json').exists():
 old=read(out/'selection.json');assert old['chosen']==chosen
else:(out/'selection.json').write_text(json.dumps(selection,indent=2))
assert shutil.disk_usage(B).free>512*2**20
free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>18000
evaluator=out/'evaluate.py';shutil.copy2(B/'evaluate_perceptual_fsm.py',evaluator)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1')
verify=[policy,str(evaluator),'--checkpoint',chosen['checkpoint'],'--out',str(out/'integration'),
 '--verify-only','--verify-replay-dir',chosen['replay']]
result=subprocess.run(verify,env=env,capture_output=True,text=True,timeout=90)
(out/'integration.log').write_text(result.stdout+result.stderr)
assert result.returncode==0,result.stdout+result.stderr
assert read(out/'integration/verification.json')['action_probability_replay_max_error']==0
supervise=out/'supervise.py'
supervise.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent
chosen=json.loads((out/'selection.json').read_text())['chosen']
def save(value):
 temp=out/'supervisor.tmp';temp.write_text(json.dumps(value,indent=2));os.replace(temp,out/'supervisor.json')
with (out/'driver.log').open('ab') as log:
 child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'evaluate.py'),'--checkpoint',chosen['checkpoint'],'--out',str(out/'fixed80')],stdout=log,stderr=log,cwd=out)
 state={'pid':child.pid,'supervisor_pid':os.getpid(),'state':'running','time':time.time()};save(state)
 code=child.wait();state.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time());save(state)
''')
with (out/'supervisor.log').open('ab') as log:
 child=subprocess.Popen(['/usr/bin/python3',str(supervise)],env=env,cwd=out,stdout=log,stderr=log,start_new_session=True)
record={'time':time.time(),'supervisor_pid':child.pid,'gpu':0,'chosen':chosen,
 'evaluator_sha256':hashlib.sha256(evaluator.read_bytes()).hexdigest(),'verification':read(out/'integration/verification.json'),
 'original_fixed80':True,'automatic_restart':False}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
