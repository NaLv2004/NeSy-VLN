import json,time,hashlib,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'run_rollout_proof_v8r1'
def read(path):return json.loads(path.read_text()) if path.exists() else None
result=dict(time=time.time(),status=read(out/'status.json'),launcher=read(out/'launcher.json'),checkpoint=read(out/'latest.COMMITTED.json'))
proc=Path('/proc')/str(result['launcher']['pid']);result['process_alive']=proc.exists()
if proc.exists():
    result['process_command']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    result['process_cwd']=str((proc/'cwd').resolve())
    result['process_identity_verified']=str(root/'src_rollout_proof_v8r1/train_rollout_proof.py') in result['process_command'] and result['process_cwd']==str(root/'src_rollout_proof_v8r1')
if result['checkpoint']:
    actual=hashlib.sha256((out/result['checkpoint']['path']).read_bytes()).hexdigest()
    result['committed_hash_verified']=actual==result['checkpoint']['sha256'] and read(out/'latest.COMMITTED.json')==result['checkpoint']
rows=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()[-100:]] if (out/'train.jsonl').exists() else []
result['recent']=dict(count=len(rows),mean_seconds=sum(r['seconds'] for r in rows)/max(len(rows),1))
result['validation']=[json.loads(line) for line in (out/'validation.jsonl').read_text().splitlines()] if (out/'validation.jsonl').exists() else []
result['driver_tail']=(out/'driver.log').read_text(errors='replace')[-2500:] if (out/'driver.log').exists() else ''
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
result['previous_terminal_status']=read(root/'run_visual_memory_subgoal_v6/status.json')
print(json.dumps(result))
