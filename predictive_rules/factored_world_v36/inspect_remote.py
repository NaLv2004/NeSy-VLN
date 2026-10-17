import json
from pathlib import Path
import subprocess
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
def read(p):return json.loads(p.read_text())
manifest=read(ROOT/'run_action_grounded_v33/manifest.json')
source=Path(manifest['data']['source'])
rows=read(source/'training_rows.json')
raw=ROOT/'raw_successor_features_v31'
metas=sorted(raw.glob('counterfactual_*/*.json'))
print(json.dumps(dict(manifest={k:manifest[k] for k in ['data','frozen_vision']},
    demonstration_rows=len(rows),row_example=rows[0],raw_meta_count=len(metas),
    raw_meta_example=read(metas[0]),raw_path=str(metas[0]),
    feature_manifest=read(ROOT/'prequant_visual_features_v18/manifest.json'),
    disk=subprocess.check_output(['df','-h',str(ROOT)],text=True),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))))
