import json,time,hashlib,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'run_visual_graph_subgoal_v5'
def read(path):return json.loads(path.read_text()) if path.exists() else None
result={'time':time.time(),'status':read(out/'status.json'),'launcher':read(out/'launcher.json'),'checkpoint':read(out/'latest.COMMITTED.json')}
launcher=result['launcher'];proc=Path('/proc')/str(launcher['pid'])
result['process_alive']=proc.exists()
if proc.exists():
    result['process_command']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    result['process_cwd']=str((proc/'cwd').resolve())
    result['process_identity_verified']=str(root/'src_visual_graph_subgoal_v5/train_visual_graph_subgoal.py') in result['process_command'] and result['process_cwd']==str(root/'src_visual_graph_subgoal_v5')
if result['checkpoint']:
    actual=hashlib.sha256((out/result['checkpoint']['path']).read_bytes()).hexdigest()
    result['committed_hash_verified']=actual==result['checkpoint']['sha256'] and read(out/'latest.COMMITTED.json')==result['checkpoint']
rows=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()[-100:]] if (out/'train.jsonl').exists() else []
result['recent']={'count':len(rows),'mean_seconds':sum(r['seconds'] for r in rows)/max(len(rows),1)}
result['validation']=[json.loads(line) for line in (out/'validation.jsonl').read_text().splitlines()] if (out/'validation.jsonl').exists() else []
result['driver_tail']=(out/'driver.log').read_text(errors='replace')[-1500:] if (out/'driver.log').exists() else ''
result['previous_retired']=read(root/'run_graph_subgoal_v4/retired.json')
result['previous_final_development']=read(root/'run_graph_subgoal_v4/development_step_003600/summary.json')
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(result))
