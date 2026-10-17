import hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');record=json.loads((base/'storage_hold_20261011.json').read_text())
def read(path):return json.loads(path.read_text()) if path.exists() else None
def live(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for relative,old in record['verified_commits'].items():
    out=base/relative;status=read(out/'status.json');supervisor=read(out/'supervisor.json');commit=read(out/'latest/COMMITTED')
    workers=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name==str(os.getpid()):continue
        try:cmd=path.read_bytes().replace(b'\0',b' ').decode()
        except OSError:continue
        if 'scripts/train.py --config' in cmd and str(out) in cmd and 'torch.distributed.run' not in cmd:
            workers.append({'pid':int(path.parent.name),'cmd':cmd,'stat':(path.parent/'stat').read_text().split(') ')[1].split()[0]})
    integrity=hashlib.sha256((out/'latest'/commit['checkpoint_file']).read_bytes()).hexdigest()==commit['checkpoint_sha256']
    jobs[relative]={'status':status,'supervisor':supervisor,'live_launcher':live((supervisor or {}).get('pid')),
        'workers':workers,'commit':commit,'commit_sha_verified':integrity,'advanced_from_before_hold':commit['epoch']>old['epoch'],
        'storage_wait':read(out/'latest/WAITING_FOR_STORAGE.json')}
summary={'time':time.time(),'free_gib':shutil.disk_usage(base).free/2**30,'jobs':jobs,
    'all_actual_updates_recovered':all(j['commit_sha_verified'] and j['advanced_from_before_hold'] and j['live_launcher']
        and len(j['workers'])==j['status']['world_size'] and all(w['stat'] not in ['T','Z'] for w in j['workers']) for j in jobs.values())}
(base/'storage_recovery_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
