import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=json.loads((root/'configs/navigation_v3.json').read_text())
base.update(root='/home/user/xjy/nesy_vla_20261010/discrete_v4',discrete_reasoning=True)
base['loss']['symbol_usage']=.2
for name,backend in [('discrete_vlm','vlm'),('discrete_gru','symbolic_gru')]:
    cfg=dict(base,architecture=name+'_v4',generator_backend=backend,generator_hidden=64)
    (root/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')

