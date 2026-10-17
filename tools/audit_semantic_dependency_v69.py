import json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/semantic_component_v69');r=json.loads((b/'aligned_text/result.json').read_text())
out={'time':time.time(),'scope':'Only saved component predictions; no navigation rerun','updates':[]}
for m in r['measurements']:
 entry={'update':m['update'],'audit_accuracy':m['audit']['accuracy']}
 for key in ['audit_text_erased','audit_text_shuffled']:
  entry[key]={'accuracy':m[key]['accuracy'],'changed_predictions':sum(a!=b for a,b in zip(m['audit']['predictions'],m[key]['predictions'])),'total':len(m['audit']['predictions'])}
 out['updates'].append(entry)
(b/'semantic_dependency_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
