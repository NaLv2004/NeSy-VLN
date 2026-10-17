"""Create an immutable, anonymous-only data collector and run contract checks."""
import hashlib,json,shlex
from pathlib import Path
from predictive_remote import connect
from deploy_planning_alignment import run
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
SOURCE=ROOT+'/src_anonymous_collection_v1'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
local=Path(__file__).resolve().parents[1]/'predictive_rules'
c=connect()
try:
    run(c,'test ! -f '+SOURCE+'/VERIFIED.json')
    files=sorted(p for p in local.rglob('*.py') if '__pycache__' not in p.parts and 'results' not in p.parts)
    run(c,'mkdir -p '+' '.join(shlex.quote(d) for d in sorted({SOURCE+'/'+p.relative_to(local).parent.as_posix() for p in files})))
    hashes={}
    with c.open_sftp() as ftp:
        for p in files:
            relative=p.relative_to(local).as_posix();ftp.put(str(p),SOURCE+'/'+relative)
            hashes[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
        with ftp.open(SOURCE+'/SOURCE_FILES.json','w') as stream:stream.write(json.dumps(hashes,indent=2))
    run(c,'cd '+SOURCE+' && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 '+PYTHON+' -m unittest test_anonymous -v')
    with c.open_sftp() as ftp:
        with ftp.open(SOURCE+'/VERIFIED.json','w') as stream:stream.write(json.dumps(dict(contract_tests=8,source_hashes=hashes),indent=2))
    print('Immutable collector source and terminal-horizon checks passed.')
finally:c.close()
