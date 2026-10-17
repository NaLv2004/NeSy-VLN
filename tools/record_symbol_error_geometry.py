"""Preserve the completed v11 latent error audit, without changing a live run."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'predictive_rules'
audit=json.loads((root/'evidence/v11_symbol_error_geometry.json').read_text())
progress=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
progress['v11_symbol_error_geometry']={k:v for k,v in audit.items() if k!='rows'}
progress['v11_symbol_error_geometry']['interpretation']='Errors are not mainly neighboring labels: median observed-posterior rank56/128 and only60/362 predicted symbols in top10. These are latent diagnostics, not physical distances or navigation success.'
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
path=root/'README.md'
text=path.read_text(encoding='utf-8')
marker='## Saved v11 runtime state-error geometry'
if marker not in text:
    text+='\n'+marker+'\n\n'
    text+='The completed runtime-rule audit uses the actual saved programs, with later visual encodings used only as labels. Of362 executed steps,18 matched exactly,41 had the predicted state among the observed posterior top5, and60 among top10. Median rank was56 of128. In320steps the inferred symbol did not occur in the observed window. This does not measure physical distance, but rejects the explanation that the poor exact accuracy is mainly adjacent-state ties. No policy feedback, new navigation evaluation or live training modification was used.\n'
    path.write_text(text,encoding='utf-8')
print(json.dumps(progress['v11_symbol_error_geometry']))
