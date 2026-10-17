"""Finite collection -> verified-contract -> training dependency, no retry loop."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_hybrid_replay_v54';DATA=ROOT/'onpolicy_successors_v54'
OUT=ROOT/'run_hybrid_replay_v54';DIAG=ROOT/'diagnostic_hybrid_replay_v54'
DIAG.mkdir(exist_ok=True)
def status(state,**extra):
    path=DIAG/'handoff.status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),time=time.time(),**extra),indent=2));tmp.replace(path)
try:
    launcher=json.loads((DATA/'launcher.json').read_text());pid=launcher['pid']
    path=Path(f'/proc/{pid}/cmdline')
    status('waiting_for_specific_collector',collector_pid=pid)
    while path.exists() and path.read_bytes():
        command=path.read_bytes().decode(errors='replace')
        assert str(SOURCE/'coordinate_collection.py') in command,'Collector PID reused'
        time.sleep(30)
    state=json.loads((DATA/'status.json').read_text())
    assert state['state']=='complete' and state['completed']==120 and state['exit_codes']==[0,0],state
    contract=json.loads((DIAG/'contract.json').read_text())
    for key in ('actual_optimizer_step','all_parameter_gradients_finite','independent_scalar_and_batched_hybrid_memory_agree',
                'future_images_do_not_change_context','no_extra_memory_grounding_at_runtime'):
        assert contract[key],key
    commit=json.loads((DATA/'source.COMMITTED.json').read_text())
    assert hashlib.sha256((DATA/'source.pt').read_bytes()).hexdigest()==commit['sha256']==contract['source_commit']['sha256']
    assert shutil.disk_usage(ROOT).free>350_000_000,'Insufficient disk headroom for committed training'
    raw=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(line.split(',')[1])<500 for line in raw.splitlines()),'GPU6/7 still owned by another live process'
    assert not (OUT/'supervisor.json').exists(),'Do not duplicate training supervisor'
    OUT.mkdir(exist_ok=True)
    log=(OUT/'supervisor.log').open('ab',buffering=0)
    process=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status('training_dispatched',supervisor_pid=process.pid,source_commit=commit,
        collection=state,run=str(OUT),gpus=[6,7],planned_updates=2000,eval_every=500,development_episodes=80)
except Exception as error:
    status('needs_repair',error=repr(error));raise
