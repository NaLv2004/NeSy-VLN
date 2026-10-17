"""Wait on the exact predecessor process exit, then launch one owned run."""
import hashlib,json,os,select,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OLD=ROOT/'run_hybrid_replay_v54';SOURCE=ROOT/'src_temporal_symbols_v55'
DIAG=ROOT/'diagnostic_temporal_symbols_v55';OUT=ROOT/'run_temporal_symbols_v55'
DIAG.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text())
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def status(state,**extra):write(DIAG/'handoff.status.json',dict(state=state,pid=os.getpid(),time=time.time(),**extra))
try:
    sv=read(OLD/'supervisor.json');pid=sv['pid'];cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists() and cmd.read_bytes():
        assert str(ROOT/'src_hybrid_replay_v54/supervisor.py') in cmd.read_bytes().decode(errors='replace')
        status('waiting_for_exact_predecessor_exit',predecessor_pid=pid)
        try:handle=os.pidfd_open(pid)
        except ProcessLookupError:handle=None
        if handle is not None:
            try:select.select([handle],[],[])
            finally:os.close(handle)
    sv=read(OLD/'supervisor.json');terminal=read(OLD/'status.json')
    assert sv['state']=='finished' and sv['exit_code']==0 and terminal['update']==2000,sv
    for rank in (0,1):
        rp=read(OLD/f'rank{rank}.status.json')['pid'];path=Path(f'/proc/{rp}/cmdline')
        assert not path.exists() or not path.read_bytes(),'Predecessor rank still live'
    development=[json.loads(s) for s in (OLD/'development.jsonl').read_text().splitlines() if s.strip()]
    assert {r['update'] for r in development}=={500,1000,1500,2000} and all(r['episodes']==80 for r in development)
    if max(r['successes'] for r in development)>=65:
        status('development_gate_reached_requires_finalfixed',development=development)
        raise SystemExit(0)
    contract=read(DIAG/'contract.json')
    assert all(contract[k] for k in ('real_optimizer_step','all_parameter_gradients_finite',
        'new_head_input_is_three_hard_ids','runtime_tables_exactly_preserved_before_training',
        'runtime_training_heads_forbidden','future_images_cannot_change_initial_context'))
    # Choose the better unchanged-dev80 source among the two compatible
    # current architectures. Never replace a better source with a worse one.
    candidates=[]
    for folder in (ROOT/'run_goal_aware_symbols_v52',OLD):
        commit=read(folder/'best.COMMITTED.json');summary=read(folder/f'dev_{commit["update"]:06d}/summary.json')
        assert summary['episodes']==80
        candidates.append(dict(path=str(folder/commit['filename']),commit=commit,successes=summary['successes']))
    chosen=max(candidates,key=lambda x:x['successes'])
    source=Path(chosen['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==chosen['commit']['sha256']
    if not (DIAG/'source.pt').exists():os.link(source,DIAG/'source.pt')
    assert hashlib.sha256((DIAG/'source.pt').read_bytes()).hexdigest()==chosen['commit']['sha256']
    write(DIAG/'source.COMMITTED.json',dict(chosen['commit'],source_path=str(source),source_successes=chosen['successes']))
    assert shutil.disk_usage(ROOT).free>1_100_000_000,'Insufficient storage headroom'
    raw=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(line.split(',')[1])<500 for line in raw.splitlines()),'GPU6/7 has another owner'
    if not (SOURCE/'predictive_nesy').exists():
        shutil.copytree(ROOT/'src_hybrid_replay_v54/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
    assert not (OUT/'supervisor.json').exists(),'Training already dispatched; inspect its exact handle'
    OUT.mkdir(exist_ok=True);log=(OUT/'supervisor.log').open('ab',buffering=0)
    process=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status('training_dispatched',supervisor_pid=process.pid,source=chosen,source_candidates=candidates,
        predecessor_terminal=sv,predecessor_development=development,gpus=[6,7],planned_updates=2000,
        eval_every=500,development_episodes=80)
except Exception as error:
    status('needs_repair',error=repr(error));raise
