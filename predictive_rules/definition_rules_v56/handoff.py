"""Event-driven handoff on the exact v55 supervisor, preserving every result."""
import hashlib,json,os,select,shutil,subprocess,time
from pathlib import Path

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OLD=ROOT/'run_temporal_symbols_v55';SOURCE=ROOT/'src_definition_rules_v56'
DIAG=ROOT/'diagnostic_definition_rules_v56';OUT=ROOT/'run_definition_rules_v56'

def read(path):return json.loads(path.read_text())
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def status(state,**kwargs):write(DIAG/'handoff.status.json',dict(state=state,pid=os.getpid(),time=time.time(),**kwargs))

try:
    sv=read(OLD/'supervisor.json');pid=sv['pid'];path=Path(f'/proc/{pid}/cmdline')
    if path.exists() and path.read_bytes():
        assert str(ROOT/'src_temporal_symbols_v55/supervisor.py') in path.read_bytes().decode(errors='replace')
        status('waiting_for_exact_predecessor_exit',predecessor_pid=pid)
        try:handle=os.pidfd_open(pid)
        except ProcessLookupError:handle=None
        if handle is not None:
            try:select.select([handle],[],[])
            finally:os.close(handle)
    sv=read(OLD/'supervisor.json');terminal=read(OLD/'status.json')
    assert sv['state']=='finished' and sv['exit_code']==0 and terminal['update']==2000,sv
    for rank in (0,1):
        p=Path(f'/proc/{read(OLD/f"rank{rank}.status.json")["pid"]}/cmdline')
        assert not p.exists() or not p.read_bytes(),'Previous rank still alive'
    dev=[json.loads(s) for s in (OLD/'development.jsonl').read_text().splitlines() if s.strip()]
    assert {r['update'] for r in dev}=={500,1000,1500,2000} and all(r['episodes']==80 for r in dev)
    if max(r['successes'] for r in dev)>=65:
        status('development_gate_reached_requires_finalfixed',predecessor_development=dev)
        raise SystemExit(0)
    contract=read(DIAG/'contract.json');pilot=read(DIAG/'result.json')
    assert contract['real_optimizer_step'] and contract['all_trainable_gradients_finite']
    assert contract['compiled_row_matches_forward'] and contract['pure_numpy_planner_verified']
    assert contract['rename_equivariance_error']<2e-5 and contract['planner_neural_calls']==0
    assert pilot['training_only'] and not pilot['original_dev80_used'] and not pilot['finalfixed80_used']
    initial=ROOT/'diagnostic_temporal_symbols_v55/source.pt'
    commit=read(initial.with_suffix('.COMMITTED.json'))
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==commit['sha256']==pilot['source']['sha256']
    if not (DIAG/'source.pt').exists():os.link(initial,DIAG/'source.pt')
    assert hashlib.sha256((DIAG/'source.pt').read_bytes()).hexdigest()==commit['sha256']
    write(DIAG/'source.COMMITTED.json',dict(commit,source_path=str(initial)))
    assert shutil.disk_usage(ROOT).free>1_100_000_000,'Need storage before launch'
    gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(line.split(',')[1])<500 for line in gpu.splitlines()),'GPU6/7 not free'
    if not (SOURCE/'predictive_nesy').exists():
        shutil.copytree(ROOT/'src_temporal_symbols_v55/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
    assert not (OUT/'supervisor.json').exists(),'Already launched; inspect the existing handle'
    OUT.mkdir(exist_ok=True)
    with (OUT/'supervisor.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status('training_dispatched',supervisor_pid=process.pid,source=commit,
        predecessor_terminal=sv,predecessor_development=dev,gpus=[6,7],planned_updates=2000,
        eval_every=500,development_episodes=80,initialization='v52 pinned source plus probe-fitted definition rule; full joint categorical losses')
except Exception as error:
    status('needs_repair',error=repr(error));raise
