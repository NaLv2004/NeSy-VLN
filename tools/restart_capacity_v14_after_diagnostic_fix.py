import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/goal_condition_capacity_v14')
manifest=root/'processes.json';records=json.loads(manifest.read_text());new=[]
for record in records:
    if record['name'].endswith('_retry1'):continue
    proc=Path(f"/proc/{record['pid']}/cmdline")
    assert not proc.exists() or not proc.read_bytes(),'Prior process remains live'
    name=record['name']+'_retry1';out=root/name;assert not out.exists();out.mkdir()
    command=list(record['command']);command[command.index('--out')+1]=str(out)
    with (out/'probe.log').open('ab') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    item={'name':name,'pid':child.pid,'command':command,'time':time.time(),'supersedes_failed':record['name'],
          'fix':'Record unsupported counterfactuals as abstentions; accept verified RGB encoder input channels'}
    (out/'process.json').write_text(json.dumps(item,indent=2));new.append(item)
manifest.write_text(json.dumps(records+new,indent=2));print(json.dumps(new,indent=2))
