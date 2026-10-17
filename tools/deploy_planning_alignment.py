"""Deploy one immutable source snapshot; allocate only newly authorized idle GPUs."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path
from predictive_remote import connect

ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
SOURCE=ROOT+'/src_alignment_v4'


def run(client,command,timeout=180):
    _,out,err=client.exec_command(command,timeout=timeout)
    result=out.read().decode()+err.read().decode()
    print(result,flush=True)
    if out.channel.recv_exit_status():raise RuntimeError('Remote operation failed')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upload',action='store_true');p.add_argument('--collect',action='store_true')
    p.add_argument('--train',action='store_true');a=p.parse_args();c=connect()
    local=Path(__file__).resolve().parents[1]/'predictive_rules'
    try:
        if a.upload:
            # Never change code after either trainer or collector was launched.
            run(c,'test ! -f '+SOURCE+'/LAUNCHED.json')
            files=sorted(p for p in local.rglob('*.py') if '__pycache__' not in p.parts and 'results' not in p.parts)
            dirs=sorted({SOURCE+'/'+p.relative_to(local).parent.as_posix() for p in files})
            run(c,'mkdir -p '+' '.join(map(shlex.quote,dirs)))
            hashes={}
            with c.open_sftp() as ftp:
                for file in files:
                    relative=file.relative_to(local).as_posix()
                    ftp.put(str(file),SOURCE+'/'+relative)
                    hashes[relative]=hashlib.sha256(file.read_bytes()).hexdigest()
                with ftp.open(SOURCE+'/SOURCE_FILES.json','w') as stream:stream.write(json.dumps(hashes,indent=2))
            run(c,'cd '+SOURCE+' && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 '+PYTHON+
                ' -m unittest test_contract test_memory test_counterfactual test_planning_alignment -v')
        if not (a.collect or a.train):return
        script=f'''import hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path({ROOT!r});source=pathlib.Path({SOURCE!r});run=root/'run_alignment_v4';run.mkdir(exist_ok=True)
data=root/'counterfactual_own_policy_v1';data.mkdir(exist_ok=True)
snapshot=run/'initial_outcomes_v3.pt'
if not snapshot.exists():
    old=root/'run_outcomes_v3';commit=json.loads((old/'best_development.COMMITTED.json').read_text())
    temporary=snapshot.with_suffix('.tmp')
    shutil.copyfile(old/'best_development.pt',temporary)
    digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest!=commit['sha256']:raise RuntimeError('Source changed during copy; inspect before retrying')
    os.replace(temporary,snapshot);(run/'initial_source.json').write_text(json.dumps(commit,indent=2))
def idle(gpu):
    used=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>200:raise RuntimeError('GPU '+str(gpu)+' is no longer idle; preserve existing processes')
def launch(cmd,gpu,folder):
    folder.mkdir(exist_ok=True);record=folder/'launcher.json'
    if record.exists():raise RuntimeError('Already launched: '+str(folder)+'; inspect its exact PID')
    idle(gpu)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    log=(folder/'driver.log').open('ab',buffering=0)
    process=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=process.pid,time=time.time(),gpu=gpu,command=cmd)
    record.write_text(json.dumps(value,indent=2));(source/'LAUNCHED.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
if {a.collect!r}:
    idle(4);idle(5)
    cmd=[{PYTHON!r},'collect_policy_outcomes.py','--checkpoint',str(snapshot),'--out',str(data),'--episodes','128','--shards','2','--shard','0','--prepare-only']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    subprocess.run(cmd,cwd=source,env=env,check=True)
    for shard,gpu in [(0,4),(1,5)]:
        cmd=[{PYTHON!r},'-u','collect_policy_outcomes.py','--checkpoint',str(snapshot),'--out',str(data),'--episodes','128','--shards','2','--shard',str(shard)]
        launch(cmd,gpu,data/('shard_'+str(shard)))
if {a.train!r}:
    pilot=json.loads((root/'pilot_planning_alignment/summary.json').read_text())
    before=pilot['before']['other_training_windows'];after=pilot['after']['other_training_windows']
    if after['beam_first_action_match']<=before['beam_first_action_match']:
        raise RuntimeError('Pilot has not improved plan agreement; inspect before training')
    if after['grounding_accuracy']<before['grounding_accuracy']-.05 or after['open_loop_observed_state_accuracy']<before['open_loop_observed_state_accuracy']-.05:
        raise RuntimeError('Pilot degraded state prediction; inspect before training')
    cmd=[{PYTHON!r},'-u','train_memory.py','--out',str(run),'--updates','6000','--batch-size','32','--eval-every','400',
         '--outcomes',str(root/'counterfactual_training_v1'),'--extra-outcomes',str(data),'--cf-batch-size','8',
         '--warm-start',str(snapshot),'--planning-weight','5','--planning-batch-size','4','--planning-beam-width','32','--resume']
    launch(cmd,3,run)
'''
        run(c,"python3 - <<'NEW_RESOURCES'\n"+script+"\nNEW_RESOURCES")
    finally:c.close()


if __name__=='__main__':main()
