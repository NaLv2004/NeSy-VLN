import json,os,signal,subprocess,time,hashlib
from pathlib import Path
parent=Path('/home/user/xjy/nesy_vla_20261010');root=parent/'symbolic_sft_v5'
old=parent/'v3/runs/full_v3'
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
for name in ['sft_plain','sft_spatial']:
    if (root/'runs'/name/'supervisor.json').exists():raise RuntimeError('New run already initialized')
    cfg=json.loads((root/'configs'/f'{name}.json').read_text())
    assert cfg['horizon']==32 and cfg['revision_interval']==10 and cfg['training_mode']=='supervised'
    assert cfg['discrete_reasoning'] and cfg['generator_backend']=='symbolic_gru'
# The continuous baseline has not improved; preserve all committed artifacts.
old.joinpath('STOP').write_text('Reallocated to strict discrete-symbol grounding experiments. All checkpoints retained.\n')
state=json.loads((old/'supervisor.json').read_text());pid=state.get('pid');proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    command=proc.read_bytes().replace(b'\0',b' ').decode()
    if str(old) not in command or 'scripts/train.py' not in command or os.getpgid(pid)!=pid:
        raise RuntimeError('Old process ownership mismatch')
    os.killpg(pid,signal.SIGTERM);deadline=time.time()+45
    while proc.exists() and time.time()<deadline:time.sleep(1)
    if proc.exists():raise RuntimeError('Old process has not exited; no new allocation started')
marker=json.loads((old/'latest/COMMITTED').read_text());file=old/'latest'/marker['checkpoint_file']
h=hashlib.sha256()
with file.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==marker['checkpoint_sha256']
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={int(a):int(b) for a,b in (row.split(',') for row in usage.strip().splitlines())}
# The two existing single-card trials still have their renderers on GPU0 until
# their current evaluation finishes. No unrelated processes are terminated.
assert used[0]<6000 and all(used[i]<1500 for i in range(1,6)),used
lineage={'time':time.time(),'stopped_run':str(old),'preserved_latest':marker,'gpu_memory_before_launch':used}
(root/'allocation.json').write_text(json.dumps(lineage,indent=2))
jobs=[]
for name,gpus in [('sft_plain','0,1,2'),('sft_spatial','3,4,5')]:
    out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
    with (out/'supervisor.log').open('ab') as log:
        child=subprocess.Popen([python,'-u','scripts/supervise.py','--out',str(out),
            '--config',f'configs/{name}.json','--gpus',gpus],cwd=root,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append({'supervisor_pid':child.pid,'gpus':gpus,'run':str(out)})
print(json.dumps({'allocation':lineage,'started':jobs},indent=2))
