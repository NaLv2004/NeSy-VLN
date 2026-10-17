import json,os,subprocess,time,hashlib,shutil
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
records=[]
for name,gpus in [('memory_dynamics_v77','2,3'),('joint_memory_dynamics_v78','6,7')]:
    out=ROOT/('run_'+name);source=ROOT/('src_'+name)
    if out.exists():
        status=json.loads((out/'supervisor.json').read_text())
        proc=Path(f"/proc/{status['pid']}/cmdline")
        assert status['state']=='needs_repair' and not(proc.exists() and proc.read_bytes())
        assert not list(out.glob('checkpoint_*.pt')) and not (out/'rank0.jsonl').exists(),'Use saved checkpoint instead'
    for gpu in gpus.split(','):
        used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert used<500,(gpu,used)
    out.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'src_expanded_dynamics_v71/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    with (out/'supervisor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(['python3','-u',str(source/'supervisor.py')],cwd=source,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(name=name,pid=child.pid,gpus=gpus,source=str(source),run=str(out),time=time.time())
    (out/'launch.json').write_text(json.dumps(record,indent=2));records.append(record)
print(json.dumps(records))
