"""Drain only owned v43 ranks, verify checkpoint, then start v48 on GPU6/7."""
import hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_contextual_symbols_v48';OUT=ROOT/'run_contextual_symbols_v48'
OLD=ROOT/'run_boundary_balanced_v43'
def cmd(pid):
    p=Path(f'/proc/{pid}/cmdline')
    try:return p.read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return ''
def main():
    assert not (OUT/'launcher.json').exists(), 'Do not duplicate existing launch'
    contract=json.loads((ROOT/'diagnostic_contextual_symbols_v48/contract.json').read_text())
    assert contract['archive_replay_exact'] and contract['future_images_cannot_affect_boundary_context']
    assert contract['all_parameter_gradients_finite'] and contract['proof_audit_passed']
    assert shutil.disk_usage(ROOT).free>1.3*1024**3
    if not (SOURCE/'predictive_nesy').exists():
        shutil.copytree(ROOT/'src_boundary_balanced_v43/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
    old=json.loads((OLD/'supervisor.json').read_text());ranks=[json.loads((OLD/f'rank{r}.status.json').read_text()) for r in (0,1)]
    active=[r['pid'] for r in ranks if 'src_boundary_balanced_v43/train.py' in cmd(r['pid'])]
    if active:
        assert len(active)==2 and 'src_boundary_balanced_v43/supervisor.py' in cmd(old['pid'])
        os.kill(old['pid'],signal.SIGSTOP)
        try:
            for pid in active:os.kill(pid,signal.SIGTERM)
            deadline=time.time()+120
            while time.time()<deadline and any(cmd(pid) for pid in active):time.sleep(1)
            if any(cmd(pid) for pid in active):raise RuntimeError('Ranks still draining; do not start replacement')
            os.kill(old['pid'],signal.SIGTERM)
        finally:os.kill(old['pid'],signal.SIGCONT)
    deadline=time.time()+20
    while time.time()<deadline and 'src_boundary_balanced_v43/' in cmd(old['pid']):time.sleep(1)
    assert 'src_boundary_balanced_v43/' not in cmd(old['pid'])
    assert all(not cmd(r['pid']) for r in ranks)
    commit=json.loads((OLD/'latest.COMMITTED.json').read_text())
    assert hashlib.sha256((OLD/commit['filename']).read_bytes()).hexdigest()==commit['sha256']
    source=ROOT/'diagnostic_sparse_support_v44/source.pt';source_commit=json.loads(source.with_suffix('.COMMITTED.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_commit['sha256']
    development=[json.loads(line) for line in (OLD/'development.jsonl').read_text().splitlines() if line.strip()]
    OUT.mkdir(exist_ok=True)
    record=dict(time=time.time(),predecessor_final=commit,predecessor_best=json.loads((OLD/'best.COMMITTED.json').read_text()),
        predecessor_development=development,source=str(source),source_commit=source_commit,contracts=contract,
        gpus=[6,7],changes='Full context-dependent anonymous observation definitions, frozen during each rule program; unchanged exact symbolic action derivation.',
        fixed_split_unchanged=True,finalfixed80_used=False,no_navigation_improvement_claim=True)
    (OUT/'launcher.json').write_text(json.dumps(record,indent=2))
    with (OUT/'supervisor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,run=str(OUT),source_commit=source_commit,predecessor_final=commit,time=time.time())),flush=True)
if __name__=='__main__':main()
