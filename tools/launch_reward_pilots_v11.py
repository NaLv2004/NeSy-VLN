import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_reward_v11')
assert not (root/'pilot_processes.json').exists()
assert '68 passed' in (root/'verification.log').read_text()
previous=json.loads((root/'integration_process.json').read_text())
assert not Path(f"/proc/{previous['pid']}/cmdline").exists()
assert json.loads((root/'integration/status.json').read_text())['state']=='complete'
row=json.loads((root/'integration/rank0.jsonl').read_text().splitlines()[-1])
assert row['losses']['ppo_ratio_max_error']<1e-4 and row['symbolic_audit']['discrete_boundary']
assert row['gradient_by_module']['logic.']>0 and row['gradient_by_module']['grounder.query.']>0
assert shutil.disk_usage(root).free>6*2**30
processes=[]
for name,gpu in [('original',6),('bounded_repeat_stop40',7)]:
    out=root/'runs'/name;out.mkdir(parents=True)
    with (out/'pilot.log').open('ab') as log:
        child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(root/f'run_{name}.py')],
            cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1'))
    processes.append({'name':name,'pid':child.pid,'gpu':gpu,'memory_fraction':.30,'out':str(out)})
(root/'pilot_processes.json').write_text(json.dumps(processes,indent=2))
manifest=json.loads((root/'experiment.json').read_text());manifest.update(tests_passed=68,
    integration={'update':row['update'],'ppo_ratio_max_error':row['losses']['ppo_ratio_max_error'],
        'discrete_boundary':row['symbolic_audit']['discrete_boundary'],'logic_gradient':row['gradient_by_module']['logic.']},
    source_files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['nesy_vla','scripts','tests','configs']
        for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts},processes=processes)
(root/'experiment.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'processes':processes,'integration':manifest['integration']}))
