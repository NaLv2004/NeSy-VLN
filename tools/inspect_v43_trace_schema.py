import json
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/spatial_automaton_v43/development')
p=next(b.glob('*/trace.json'));r=json.loads(p.read_text())
print(json.dumps({'path':str(p),'keys':list(r),'episode':r['episode'],'step0':r['steps'][0],'metrics':r.get('metrics'),'evaluation':json.loads((p.parent/'evaluation.json').read_text())},indent=2)[:9000])
