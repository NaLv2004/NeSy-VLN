import json,subprocess,shutil,time
from pathlib import Path
base=Path('/home/user/xjy');owned=base/'nesy_vla_20261010'
usage=subprocess.check_output(['du','-x','-B1','--max-depth=2',str(base)],text=True)
rows=sorted((line.split('\t',1) for line in usage.splitlines()),key=lambda row:int(row[0]),reverse=True)
record={'time':time.time(),'free_gib':shutil.disk_usage(base).free/2**30,
    'largest':[{'path':path,'gib':int(size)/2**30} for size,path in rows[:35]],
    'content_store_state':json.loads((owned/'evidence_archives/content_store/conversion.json').read_text())['state'],
    'direct_archive_state':json.loads((owned/'evidence_archives/direct_archive_status.json').read_text())['state']}
(owned/'storage_scope_20261011.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
