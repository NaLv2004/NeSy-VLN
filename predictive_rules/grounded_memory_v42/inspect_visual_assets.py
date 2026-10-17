"""Read-only visual input inventory for a possible foundation-encoder trial."""
import json
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
m=json.loads((root/'run_action_grounded_v33/manifest.json').read_text())
source=Path(m['data']['source']);rows=json.loads((source/'training_rows.json').read_text())
vision=m['frozen_vision'];encoder=Path(vision['encoder_source'])
cfg=json.loads((encoder.parent/'config.json').read_text())
result=dict(data_source=str(source),frozen_vision=vision,
    visual_configuration={k:v for k,v in cfg.items() if any(s in k.lower() for s in ('image','vision','encoder','backbone'))},
    example_row=rows[0],frames=len(rows),nearby_files=[p.name for p in source.iterdir()][:25])
print(json.dumps(result,indent=2))
