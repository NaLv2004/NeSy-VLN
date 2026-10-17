"""New schema deployment; no legacy semantic-classifier training is launched."""
import argparse,hashlib,json,shlex
from pathlib import Path
from predictive_remote import connect
from deploy_planning_alignment import run
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
SOURCE=ROOT+'/src_anonymous_v5'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--upload',action='store_true');a=parser.parse_args()
    c=connect();local=Path(__file__).resolve().parents[1]/'predictive_rules'
    try:
        if a.upload:
            run(c,'test ! -f '+SOURCE+'/LAUNCHED.json')
            files=sorted(p for p in local.rglob('*.py') if '__pycache__' not in p.parts and 'results' not in p.parts)
            run(c,'mkdir -p '+' '.join(shlex.quote(d) for d in sorted({SOURCE+'/'+p.relative_to(local).parent.as_posix() for p in files})))
            hashes={}
            with c.open_sftp() as ftp:
                for p in files:
                    relative=p.relative_to(local).as_posix();ftp.put(str(p),SOURCE+'/'+relative)
                    hashes[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
                with ftp.open(SOURCE+'/SOURCE_FILES.json','w') as stream:stream.write(json.dumps(hashes,indent=2))
            run(c,'cd '+SOURCE+' && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 '+PYTHON+' -m unittest test_anonymous -v')
        if not a.launch:return
        script=f'''import json,os,pathlib,subprocess,time
root=pathlib.Path({ROOT!r});source=pathlib.Path({SOURCE!r});out=root/'run_anonymous_v5';out.mkdir(exist_ok=True)
if (out/'launcher.json').exists():raise RuntimeError('Already launched: inspect its PID, do not duplicate')
used=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>200:raise RuntimeError('GPU3 is not idle; preserve existing process')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
cmd=[{PYTHON!r},'-u','train_anonymous.py','--out',str(out),'--initialize',str(root/'run_alignment_v4/initial_outcomes_v3.pt'),
     '--updates','6000','--batch-size','32','--eval-every','400','--outcomes',str(root/'counterfactual_training_v1'),
     '--outcomes',str(root/'counterfactual_own_policy_v1'),'--outcome-batch-size','8','--planning-batch-size','4','--planning-weight','2','--resume']
log=(out/'driver.log').open('ab',buffering=0)
p=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,gpu=3,time=time.time(),command=cmd,architecture='anonymous_generated_rules_v5')
(out/'launcher.json').write_text(json.dumps(record,indent=2));(source/'LAUNCHED.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
        run(c,"python3 - <<'ANONYMOUS_LAUNCH'\n"+script+"\nANONYMOUS_LAUNCH")
    finally:c.close()


if __name__=='__main__':main()
