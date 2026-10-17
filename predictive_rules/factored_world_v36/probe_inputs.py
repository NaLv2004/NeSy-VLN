import json, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
m=json.loads((ROOT/'run_action_grounded_v33/manifest.json').read_text())
p=next((ROOT/'raw_successor_features_v31').glob('counterfactual_*/*.npz'))
with np.load(p) as d:
    print(json.dumps({'npz':str(p),'arrays':{k:[list(d[k].shape),str(d[k].dtype)] for k in d.files},
        'actions':d['recorded_actions'].tolist(),'probe_valid':d['probe_valid'].tolist()}))
print('FEATURES',np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r').shape)
print(Path('/proc/meminfo').read_text().splitlines()[:4])
print('SRC_FILES',len(list((ROOT/'src_program_ppo_v35').rglob('*.py'))))
