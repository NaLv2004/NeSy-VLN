import json,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'src_runtime_audit_v22'
assert not out.exists()
states=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in states.splitlines()}[7]<300
status=json.loads((root/'run_separated_goal_v21/status.json').read_text());assert status['state']=='training_budget_complete'
shutil.copytree(root/'src_separated_goal_v21',out)
path=out/'replay_boundary_worker.py';s=path.read_text();assert s.count('if not 1 <= len(actions) <= 32:')==1
s=s.replace('if not 1 <= len(actions) <= 32:','if not 0 <= len(actions) <= 32:')
s=s.replace('result = dict(rgb=np.stack(frames))',"rgb = np.asarray(env.observe()['rgb'])[..., :3].astype(np.uint8)\n                    frames.append(np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR)))\n                    result = dict(rgb=np.stack(frames))")
path.write_text(s)
print(json.dumps(dict(source=str(out),gpus=states,previous_training=status)))
