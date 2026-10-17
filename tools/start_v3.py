"""Replace the unsuccessful v2 run only after committing its final snapshot."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path

parent=Path('/home/user/xjy/nesy_vla_20261010');old=parent/'v2/runs/full_v2'
root=parent/'v3';out=root/'runs/full_v3';out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
if (out/'supervisor.json').exists():
    pid=read(out/'supervisor.json').get('pid')
    proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists() and b'scripts/train.py' in proc.read_bytes():raise RuntimeError('v3 already running')
cfg=read(root/'configs/navigation_v3.json')
assert cfg['evidence_only_grounder'] and cfg['horizon']==32 and cfg['revision_interval']==10
# Stop only the verified v2 process group. Every completed update was committed.
(old/'STOP').write_text('Superseded by evidence-grounded v3; preserve checkpoints.\n')
state=read(old/'supervisor.json');pid=state.get('pid');proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    command=proc.read_bytes().replace(b'\0',b' ').decode()
    if str(old) not in command or 'scripts/train.py' not in command:raise RuntimeError('Process ownership mismatch')
    if os.getpgid(pid)!=pid:raise RuntimeError('Unexpected training process group')
    os.killpg(pid,signal.SIGTERM)
    deadline=time.time()+45
    while proc.exists() and time.time()<deadline:time.sleep(1)
    if proc.exists():raise RuntimeError('v2 has not stopped; v3 not launched')
marker=read(old/'latest/COMMITTED');checkpoint=old/'latest'/marker['checkpoint_file']
h=hashlib.sha256()
with checkpoint.open('rb') as handle:
    for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==marker['checkpoint_sha256']
manifest={'previous_run':str(old),'preserved_checkpoint':str(checkpoint),'previous_commit':marker,
          'initialization':'fresh pretrained Qwen + ImageNet, identical seed and full v2 training settings',
          'changed_component':'PredicateGrounder.query: evidence-conditioned reading without additive definition residual',
          'unchanged':['soft logic sole action selector','learned definitions and rules','every10 revision',
                       'max32 actions','frozen training/development/fixed80 split','losses','learning rates','batch limits'],
          'time':time.time()}
(out/'lineage.json').write_text(json.dumps(manifest,indent=2))
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
         HF_HUB_OFFLINE='1',TRITON_CACHE_DIR=str(parent/'v2/cache/triton'))
(out/'STOP').unlink(missing_ok=True)
with (out/'supervisor.log').open('ab') as log:
    child=subprocess.Popen([python,'-u','scripts/supervise.py','--out',str(out),
        '--config','configs/navigation_v3.json','--gpus','0,1,2,3,4,5'],cwd=root,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':child.pid,'run':str(out),'lineage':manifest},indent=2))
