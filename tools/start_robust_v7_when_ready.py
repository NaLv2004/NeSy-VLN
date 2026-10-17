"""Switch only owned v5 jobs, after their committed evaluations and data verification."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_robust_v7';dataset=base/'data/lossless_demonstrations_v7'
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
plans=[('sft_plain','robust_reference','0,1,2',811916),('sft_spatial','robust_multieffect','3,4,5',811917)]
records={}
def read(p):return json.loads(p.read_text())
def command(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
def save(state):
    temp=root/'allocation.tmp';temp.write_text(json.dumps({'time':time.time(),'pid':os.getpid(),'state':state,'jobs':records},indent=2))
    temp.replace(root/'allocation.json')
assert len(read(root/'training_verification.json')['arms'])==2
# Respect the existing long-task inspection interval before examining READY.
previous=read(dataset/'inspection.json')
time.sleep(max(0,previous['time']+600-time.time()))
save('waiting_for_dataset')
while not (dataset/'READY.json').exists():
    control=read(dataset/'controller.json')
    if not command(control['pid']):raise RuntimeError('Dataset controller exited without READY; no training changed')
    time.sleep(600)
ready=read(dataset/'READY.json');examples=read(dataset/'examples.json')
assert ready['source_episodes']==ready['successful_episodes']==1951 and ready['examples']==17560
assert ready['failed_episodes']==[] and ready['max_canonical_source_pixel_mae']==0
assert hashlib.sha256((dataset/'examples.json').read_bytes()).hexdigest()==ready['examples_sha256']
provenance=read(dataset/'source.json')
assert ready['data_provenance']==provenance
assert len(examples)==17560 and len({row['episode_id'] for row in examples})==1951
assert all('collision' in row['public_feedback'] for row in examples)
(root/'dataset_verified.json').write_text(json.dumps(ready,indent=2))
save('waiting_for_saved_evaluation_boundary')
for attempt in range(12):
    for old_name,new_name,gpus,supervisor in plans:
        if new_name in records:continue
        old=base/'symbolic_sft_v5/runs'/old_name;out=root/'runs'/new_name
        if (out/'supervisor.json').exists():raise RuntimeError('New run already exists')
        state=read(old/'status.json');step=state['update'];pid=read(old/'supervisor.json').get('pid')
        if step%10==0 and not all((old/f'evaluations/step_{step:06d}'/s/'summary.json').exists() for s in ['development','fixed80']):continue
        assert str(old) in command(pid) and 'scripts/train.py' in command(pid)
        assert str(old) in command(supervisor) and 'scripts/supervise.py' in command(supervisor)
        assert os.getpgid(pid)==pid
        os.killpg(pid,signal.SIGSTOP)
        try:
            state=read(old/'status.json');step=state['update'];marker=read(old/'latest/COMMITTED')
            safe=marker['epoch']==step and (step%10!=0 or all((old/f'evaluations/step_{step:06d}'/s/'summary.json').exists() for s in ['development','fixed80']))
            if safe:(old/'STOP').write_text('Reallocated to lossless robust grounding and generated multi-effect rule comparison; checkpoints retained.\n')
        finally:os.killpg(pid,signal.SIGCONT)
        if not safe:continue
        os.killpg(pid,signal.SIGTERM);os.kill(supervisor,signal.SIGTERM)
        deadline=time.time()+45
        while (command(pid) or command(supervisor)) and time.time()<deadline:time.sleep(1)
        if command(pid) or command(supervisor):raise RuntimeError('Previous owned processes have not exited')
        marker=read(old/'latest/COMMITTED');checkpoint=old/'latest'/marker['checkpoint_file']
        assert marker['optimizer_included'] and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==marker['checkpoint_sha256']
        out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        with (out/'supervisor.log').open('ab') as log:
            child=subprocess.Popen([python,'-u','scripts/supervise.py','--out',str(out),'--config',f'configs/{new_name}.json','--gpus',gpus],
                cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records[new_name]={'old_run':str(old),'preserved_final':marker,'gpus':gpus,'supervisor_pid':child.pid,'time':time.time()}
        save('allocating');print(json.dumps(records[new_name]),flush=True)
    if len(records)==2:save('complete');break
    time.sleep(600)
else:raise RuntimeError('No safe allocation boundary found')
