"""Wait for the actual collector process exit, audit durable data, start training."""
import hashlib,json,os,select,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
DATA=ROOT/'onpolicy_successors_v60';DIAG=ROOT/'diagnostic_onpolicy_refresh_v60'
SOURCE=ROOT/'src_onpolicy_refresh_v60_training';OUT=ROOT/'run_onpolicy_refresh_v60'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def read(p):return json.loads(p.read_text())
def save(x):
    x.update(pid=os.getpid(),time=time.time());p=DIAG/'handoff.status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
    collector=read(DATA/'launcher.json')['pid']
    proc=Path(f'/proc/{collector}/cmdline')
    if proc.exists() and proc.read_bytes():
        assert b'coordinate_collection.py' in proc.read_bytes() and b'src_onpolicy_refresh_v60/' in proc.read_bytes()
        fd=os.pidfd_open(collector)
        save(dict(state='waiting_collector_exit',collector_pid=collector,event_wait=True))
        try:select.select([fd],[],[])
        finally:os.close(fd)
    status=read(DATA/'status.json');assert status['state']=='complete' and status['exit_codes']==[0,0]
    info=read(DATA/'manifest.json');assert info['training_only'] and info['no_goal_or_imitation_supervision']
    assert info['max_actions']==32 and info['source_commit']['sha256']=='58abc7f6a46587f2312412b1848c1e84dc64a374ba725b81f51dd8c0bca880bf'
    files=sorted((DATA/'counterfactual_onpolicy').glob('*.json'))
    assert len(files)==120 and {read(p)['episode'] for p in files}==set(info['episode_ids'])
    assert len(info['train_layouts'])==20
    positions=executed=root_no_proof=0
    for p in files:
        r=read(p);npz=p.with_suffix('.npz')
        assert r['scene'] in info['train_layouts'] and r['training_only'] and not r['goal_supervision'] and not r['probe_labels_available_to_policy']
        assert hashlib.sha256(npz.read_bytes()).hexdigest()==r['output_sha256']
        assert r['source_checkpoint_sha256']==info['source_commit']['sha256'] and 1<=r['positions']<=32
        assert all(s['proof']['found'] and s['action']==s['proof']['actions'][0] for s in r['steps'])
        assert all(b['start']-a['start']>=5 for a,b in zip(r['programs'],r['programs'][1:]))
        positions+=r['positions'];executed+=r['executed_actions'];root_no_proof+=r['executed_actions']==0
    audit=dict(time=time.time(),episodes=len(files),positions=positions,measured_successors=17*positions,executed_actions=executed,
        root_no_proof_but_probed=root_no_proof,training_only=True,all_checksums_verified=True,
        all_executed_actions_have_symbolic_proofs=True,source_commit=info['source_commit'])
    (DIAG/'collection_audit.json').write_text(json.dumps(audit,indent=2))
    save(dict(state='checking_training_contract',collection=audit))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (DIAG/'contract.log').open('ab',buffering=0) as log:
        subprocess.run([PYTHON,'-u',str(SOURCE/'check_contract.py')],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    contract=read(DIAG/'contract.json');assert contract['real_optimizer_step'] and contract['planner_neural_calls']==0
    assert shutil.disk_usage(ROOT).free>850_000_000
    assert not subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not (OUT/'supervisor.json').exists(),'Never duplicate the staged run'
    OUT.mkdir(exist_ok=True)
    with (OUT/'supervisor.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(dict(state='training_dispatched',supervisor_pid=p.pid,collection=audit,contract=contract,source=info['source_commit']))
try:main()
except Exception as e:
    save(dict(state='needs_repair',error=repr(e),traceback=traceback.format_exc()));raise
