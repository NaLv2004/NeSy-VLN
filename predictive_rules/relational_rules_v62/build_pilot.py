"""Reuse the audited fixed-target dataset construction, then compare two heads."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE.parent/'ensemble_rules_v61/probe.py').read_text(encoding='utf-8')
prefix=source[:source.index('    heads=[copy.deepcopy(model.definition_rule)')]
prefix=prefix.replace('diagnostic_ensemble_rules_v61','diagnostic_relational_rules_v62')
prefix=prefix.replace('import copy,json,os,random,select,sys,time,traceback','import copy,json,os,random,select,sys,time,traceback\nfrom relational_rule import RelationalRule')
prefix='"""Fixed-label action-relation pilot. Same96/24 training episode split as v61."""\n'+prefix[source.index('import copy'):]
suffix=(HERE/'pilot_body.txt').read_text(encoding='utf-8')
(HERE/'probe.py').write_text(prefix+suffix,encoding='utf-8')
print('Built fixed-target two-head pilot')
