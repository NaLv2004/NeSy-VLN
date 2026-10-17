"""Start successor only after verifying the exact old trainer is terminal."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
src=ROOT/'src_goal_aware_symbols_v52';out=ROOT/'run_goal_aware_symbols_v52';old=ROOT/'run_contextual_symbols_v48'
assert not (out/'launcher.json').exists(),'Inspect the recorded launch, do not duplicate'
contract=json.loads((ROOT/'diagnostic_goal_aware_symbols_v52/contract.json').read_text())
for key in ('initial_v48_context_and_codes_exact','future_images_cannot_affect_boundary_context','all_parameter_gradients_finite',
    'task_index_permutation_equivariant','proof_audit_passed','archive_replay_exact','no_neural_action_head_added'):
    assert contract[key],key
status=json.loads((old/'status.json').read_text());supervisor=json.loads((old/'supervisor.json').read_text())
assert status['state']=='finished' and status['update']==3000
assert supervisor['state']=='finished' and supervisor['exit_code']==0
pids=[json.loads((old/f'rank{rank}.status.json').read_text())['pid'] for rank in (0,1)]
pids.append(supervisor['pid'])
for pid in pids:
    cmd=Path(f'/proc/{pid}/cmdline')
    assert not cmd.exists() or not cmd.read_bytes(),('Old process remains alive',pid)
final=json.loads((old/'latest.COMMITTED.json').read_text())
assert hashlib.sha256((old/final['filename']).read_bytes()).hexdigest()==final['sha256']
source=ROOT/'diagnostic_goal_aware_symbols_v52/source.pt'
commit=json.loads(source.with_suffix('.COMMITTED.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==commit['sha256']
development=[json.loads(line) for line in (old/'development.jsonl').read_text().splitlines() if line.strip()]
assert {r['update'] for r in development}=={500,1000,1500,2000,2500,3000}
assert all(r['episodes']==80 for r in development)
assert shutil.disk_usage(ROOT).free>.8*1024**3
out.mkdir(exist_ok=True)
record=dict(time=time.time(),predecessor_final=final,predecessor_development=development,predecessor_status=status,
    predecessor_processes_confirmed_terminal=pids,source_commit=commit,contract=contract,
    gpus=[6,7],all_other_processes_untouched=True,source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.py')},
    primary_change='Task-conditioned generated anonymous symbol definitions and matching rule tables; current/future labels use the same fixed basis',
    numerical_repair='Normalized float64 success/failure Bellman arithmetic; same expected-reachability objective, no neural action head',
    training='Fresh AdamW with representation2e-5 and other1e-4,128 demonstrations plus64 all-action branches per GPU',
    evaluation='Original complete dev80 every500updates; finalfixed80 untouched',planned_updates=3000,
    no_navigation_improvement_claim=True)
with (out/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(src/'supervisor.py')],cwd=src,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record['pid']=child.pid;(out/'launcher.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(pid=child.pid,time=record['time'],run=str(out),predecessor_final=final,
    predecessor_development=[dict(update=r['update'],successes=r['successes'],episodes=r['episodes']) for r in development])))
