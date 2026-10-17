"""Deploy the terminal-budget/data-aggregation correction in a separate run."""
import hashlib,json,shlex
from pathlib import Path
from predictive_remote import connect
from deploy_planning_alignment import run
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
SOURCE=ROOT+'/src_anonymous_v6'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
local=Path(__file__).resolve().parents[1]/'predictive_rules'
c=connect()
try:
    run(c,'test ! -f '+SOURCE+'/LAUNCHED.json')
    files=sorted(p for p in local.rglob('*.py') if '__pycache__' not in p.parts and 'results' not in p.parts)
    run(c,'mkdir -p '+' '.join(shlex.quote(d) for d in sorted({SOURCE+'/'+p.relative_to(local).parent.as_posix() for p in files})))
    hashes={}
    with c.open_sftp() as ftp:
        for p in files:
            relative=p.relative_to(local).as_posix();ftp.put(str(p),SOURCE+'/'+relative)
            hashes[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
        with ftp.open(SOURCE+'/SOURCE_FILES.json','w') as stream:stream.write(json.dumps(hashes,indent=2))
    run(c,'cd '+SOURCE+' && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 '+PYTHON+' -m unittest test_anonymous test_anonymous_corpus -v')
    script=f'''import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path({ROOT!r});source=pathlib.Path({SOURCE!r});out=root/'run_anonymous_v6';out.mkdir(exist_ok=True)
if (out/'launcher.json').exists():raise RuntimeError('Already launched; inspect its PID rather than duplicate')
used=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>200:raise RuntimeError('GPU7 is occupied; preserve all existing processes')
parent=root/'run_anonymous_v5/best_development.pt'
for _ in range(3):
    before=json.loads(parent.with_suffix('.COMMITTED.json').read_text());data=parent.read_bytes()
    after=json.loads(parent.with_suffix('.COMMITTED.json').read_text())
    if before==after and hashlib.sha256(data).hexdigest()==before['sha256']:break
else:raise RuntimeError('Parent publication incomplete; no training process was started')
(out/'initial.pt').write_bytes(data);(out/'initial.COMMITTED.json').write_text(json.dumps(before,indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
command=[{PYTHON!r},'-u','train_anonymous.py','--out',str(out),'--initialize',str(out/'initial.pt'),'--updates','6000',
 '--batch-size','32','--eval-every','400','--outcomes',str(root/'counterfactual_training_v1'),
 '--outcomes',str(root/'counterfactual_anonymous_v5'),'--outcome-batch-size','8','--planning-batch-size','4','--planning-weight','2','--resume']
log=(out/'driver.log').open('ab',buffering=0)
p=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,gpu=7,time=time.time(),command=command,parent=before,tests_passed=10)
(out/'launcher.json').write_text(json.dumps(record,indent=2));(source/'LAUNCHED.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
    run(c,"python3 - <<'ANONYMOUS_V6'\n"+script+"\nANONYMOUS_V6")
finally:c.close()
