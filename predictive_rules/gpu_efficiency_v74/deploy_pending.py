import ast,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).parent.parent;sys.path.insert(0,str(BASE.parent/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';client=connect()
try:
    with client.open_sftp() as ftp:
        with ftp.open(ROOT+'/run_expanded_dynamics_v71/supervisor.json') as f:state=json.load(f)
        assert state['state']=='waiting_for_collection_exit',state
        # Only the waiting launcher is live; training modules have not been loaded.
        for name in ('gpu_planner.py','autobatch.py','evaluate.py','train_joint.py'):
            p=BASE/'expanded_dynamics_v71'/name;ast.parse(p.read_text());ftp.put(str(p),ROOT+'/src_expanded_dynamics_v71/'+name)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'expanded_dynamics_v71').glob('*.py') if p.name!='deploy.py'}
        with ftp.open(ROOT+'/run_expanded_dynamics_v71/source_files.json','w') as f:f.write(json.dumps(hashes,indent=2))
        for remote,local in [('gpu_efficiency_v74/benchmark.json','gpu_efficiency_v74/benchmark.json'),
                             ('run_continued_evidence_v72/result.json','continued_evidence_v72/result.json'),
                             ('run_compact_history_v73/result.json','compact_history_v73/result.json')]:
            ftp.get(ROOT+'/'+remote,str(BASE/local))
    print(json.dumps(dict(pending_v71_updated=True,waiting_pid=state['pid'],validation_workers_per_gpu=40,
                         continuous_training_source='44/80 checkpoint; full1200when collection ends',
                         v72=json.loads((BASE/'continued_evidence_v72/result.json').read_text()),
                         v73=json.loads((BASE/'compact_history_v73/result.json').read_text()))))
finally:client.close()
