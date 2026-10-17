import json
from pathlib import Path
from collections import Counter
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
m=json.loads((r/'run_action_grounded_v33/manifest.json').read_text())
p=Path(m['data']['source']);rows=json.loads((p/'training_rows.json').read_text())
print(json.dumps({'source':str(p),'data':m['data'],'row_keys':list(rows[0]),'sample':rows[0],'paired_sample':next(v for v in rows if v['episode_id'].startswith('paired_')),'files':[v.name for v in p.iterdir() if v.is_file()]}))