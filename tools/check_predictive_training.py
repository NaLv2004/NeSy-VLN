"""Inspect an existing isolated training handle without changing its state."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',default='run_grounded_aggregate_v10');a=p.parse_args()
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=(root/a.run).resolve()
assert out.parent==root and out.name.startswith('run_')
def read(path):return json.loads(path.read_text()) if path.exists() else None
launch=read(out/'launcher.json');assert launch
result=dict(time=time.time(),status=read(out/'status.json'),launcher=launch)
proc=Path('/proc')/str(launch['pid']);result['process_alive']=proc.exists()
trainer=next(x for x in launch['command'] if x.endswith('.py'))
if proc.exists():
    result['process_command']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    result['process_cwd']=str((proc/'cwd').resolve())
    result['process_identity_verified']=trainer in result['process_command'] and result['process_cwd']==launch['source']
result['checkpoint']=None;result['committed_hash_verified']=False
for _ in range(3):
    commit=read(out/'latest.COMMITTED.json')
    if commit is None:break
    digest=hashlib.sha256((out/commit['path']).read_bytes()).hexdigest()
    if commit==read(out/'latest.COMMITTED.json') and digest==commit['sha256']:
        result.update(checkpoint=commit,committed_hash_verified=True);break
rows=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()[-100:]] if (out/'train.jsonl').exists() else []
result['recent']=dict(count=len(rows),mean_seconds=sum(r['seconds'] for r in rows)/max(len(rows),1))
for key in rows[-1] if rows else []:
    if key.startswith('transition_bank_'):result['recent'][key+'_mean']=sum(r[key] for r in rows)/len(rows)
result['validation']=[json.loads(line) for line in (out/'validation.jsonl').read_text().splitlines()] if (out/'validation.jsonl').exists() else []
result['driver_tail']=(out/'driver.log').read_text(errors='replace')[-2500:] if (out/'driver.log').exists() else ''
result['evaluation_error']=read(out/'evaluation_error.json')
result['frozen_representation_verified']=read(out/'frozen_representation_verified.json')
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
manifest=read(out/'manifest.json')
if manifest:result['training_scope']={k:manifest[k] for k in ['architecture','trainable_modules','state_representation_frozen','trainable_parameters','world_model_intervention','initialization_scope','transition_sampling','transition_banks','input','output','no_continuous_state_target','no_semantic_state_labels','no_navigation_controller'] if k in manifest}
print(json.dumps(result))
