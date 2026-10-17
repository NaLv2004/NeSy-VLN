import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_goal_boundary_data_v66';OUT=ROOT/'goal_boundary_data_v66'
assert json.loads((ROOT/'run_linear_evidence_v65/supervisor.json').read_text())['state']=='finished'
OUT.mkdir(exist_ok=True);assert not (OUT/'status.json').exists()
for gpu in (6,7):
    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    assert used<500,(gpu,used)
source=json.loads((ROOT/'counterfactual_closure_v63/manifest.json').read_text())
manifest=dict(episode_ids=source['episode_ids'],training_only=True,source='goal_discrimination_v64',
    original_training_and_validation_layouts_unchanged=True,finalfixed80_used=False,
    same_train_holdout_split_as_v63=True,no_map=True,no_navigation_teacher=True,
    strategy='At first and last recorded positive position plus preceding position; if none, final decision position. Probe all17actions and one continuing second action per non-STOP first action.',
    labels='Binary existing environment STOP outcome; no state IDs or distances supplied',
    workers_per_gpu=24,cnn_batch=256,feature_schema='49x128 normalized frozen vision patches',
    output='each measured successor carries root step and 1/2 opaque action IDs')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
with (OUT/'coordinator.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'coordinate_collection.py')],cwd=SOURCE,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=p.pid,source=str(SOURCE),data=str(OUT));(OUT/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
