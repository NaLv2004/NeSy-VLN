import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_branch_v17'
probe=json.loads((base/'branch_rule_capacity_v16/completion_inspection.json').read_text())
assert all(v['state']=='complete' and v['last']['step']==800 for v in probe['results'].values())
assert not (root/'process.json').exists(),'Inspect existing process instead of starting again'
assert shutil.disk_usage(base).free>5*2**30
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(a):int(b) for a,b in (line.split(',') for line in memory.splitlines())}
assert all(free[g]>21000 for g in [6,7]),'GPUs6/7 are not free'
# Visual-only transfer did not show a convincing held-out gain. Keep perception
# initialization matched to v15; the tested change is generated branch rules.
variant='branch_current';out=root/'runs'/variant;out.mkdir(parents=True)
command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','scripts/supervise.py',
    '--config',str(root/'configs'/f'{variant}.json'),'--out',str(out),'--gpus','6,7']
with (out/'supervisor.log').open('ab') as log:
    child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
        start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
record={'time':time.time(),'supervisor_pid':child.pid,'variant':variant,'command':command,'gpus':[6,7],
        'probe_result':'All800 updates completed with total symbolic branch coverage; first-action generalization remains poor.',
        'why_full_trajectory_training':'Test coverage-preserving learned rules on the complete original demonstration corpus plus paired goals.',
        'no_success_claim':True}
(root/'process.json').write_text(json.dumps(record,indent=2))
experiment=json.loads((root/'experiment.json').read_text());experiment.update(state='launched',selected_variant=variant)
(root/'experiment.json').write_text(json.dumps(experiment,indent=2));print(json.dumps(record,indent=2))
