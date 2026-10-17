import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'token_revision_rl_v45';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect original handles; do not duplicate launch'
assert json.loads((base/'spatial_automaton_v43/status.json').read_text())['state']=='complete'
free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>16000 and shutil.disk_usage(base).free>2*1024**3
src=out/'src';src.mkdir(exist_ok=True)
shutil.copytree(base/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(base/'train_token_revision_rl_v45.py',src/'train.py')
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
for attempt in range(10):
    with (out/'driver.log').open('ab') as log:
        p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py')],cwd=out/'src',env=env,stdout=log,stderr=log)
        row={'supervisor_pid':os.getpid(),'pid':p.pid,'attempt':attempt,'time':time.time(),'state':'running'}
        (out/'supervisor.json').write_text(json.dumps(row,indent=2))
        code=p.wait()
    row.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time())
    (out/'supervisor.json').write_text(json.dumps(row,indent=2))
    if code==0:break
    time.sleep(5)
''')
with (out/'supervisor.log').open('ab') as log:
    proc=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],cwd=out,stdout=log,stderr=log,start_new_session=True)
record={'pid':proc.pid,'time':time.time(),'script':str(src/'train.py'),'supervisor':str(supervisor),'gpu':7,
        'source_program':'spatial_automaton_v43','original_train_development_fixed_split_unchanged':True}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
