import hashlib,json,os,signal,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');data=base/'data/paired_goals_v12';out=base/'paired_goal_probe_v12'
record=json.loads((out/'controller_process.json').read_text());pid=record['pid'];proc=Path(f'/proc/{pid}/cmdline')
existing=out/'probe_process.json'
if existing.exists():
    child=json.loads(existing.read_text());p=Path(f"/proc/{child['pid']}/cmdline")
    if p.exists() and b'probe_paired_goal_learning.py' in p.read_bytes():
        print(json.dumps({'already_running_probe':child}));raise SystemExit
if (out/'result.json').exists():raise RuntimeError('A prior probe result already exists; inspect it before a new launch')
if proc.exists():
    assert b'start_pair_probe_when_ready.py' in proc.read_bytes()
    status=json.loads((out/'controller_status.json').read_text());assert status['state']=='waiting_for_committed_data'
    os.kill(pid,signal.SIGTERM)
    for _ in range(20):
        if not proc.exists() or not proc.read_bytes():break
        time.sleep(.1)
    else:raise RuntimeError('Controller did not stop; refusing duplicate probe')
snapshot=out/'input_snapshot';snapshot.mkdir(exist_ok=True)
records=[json.loads(p.read_text()) for p in sorted((data/'pairs').glob('*/COMMITTED.json'))]
records=[r for r in records if r['both_successful'] and r['same_initial_rgb']]
assert sum(r['first_actions_differ'] for r in records)>=8
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for r in records:
    for branch in r['branches']:
        for row in branch['frames']:
            source=(data/row['image']).resolve();dest=(snapshot/row['image']).resolve()
            assert source.is_relative_to(data.resolve()) and dest.is_relative_to(snapshot.resolve())
            assert sha(source)==row['sha256'];dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():os.link(source,dest)
            assert sha(dest)==row['sha256']
provenance=json.loads((data/'source.json').read_text())
contents={'source':provenance,'pairs':records,'episodes':[b['episode'] for r in records for b in r['branches']],
    'examples':[f for r in records for b in r['branches'] for f in b['frames']]}
for name,value in contents.items():(snapshot/(name+'.json')).write_text(json.dumps(value,indent=2))
complete={'time':time.time(),'provenance':provenance,'pairs':len(records),'collection_continues_elsewhere':str(data)}
for name in ['pairs','examples','episodes']:complete[name+'_sha256']=sha(snapshot/(name+'.json'))
(snapshot/'COMPLETE').write_text(json.dumps(complete,indent=2))
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(base/'probe_paired_goal_learning.py'),
    '--root',str(base/'symbolic_reward_v11'),'--data',str(snapshot),
    '--checkpoint',str(base/'symbolic_reward_v11/initialization_source'),'--out',str(out),'--steps','160']
with (out/'probe.log').open('ab') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
item={'pid':child.pid,'command':cmd,'time':time.time(),'snapshot_pairs':len(records),
    'different_first_actions':sum(r['first_actions_differ'] for r in records)}
(out/'probe_process.json').write_text(json.dumps(item,indent=2))
(out/'controller_status.json').write_text(json.dumps({'state':'superseded_by_verified_snapshot_probe',
    'time':time.time(),'old_pid':pid,'new_probe_pid':child.pid,'snapshot':str(snapshot)}))
print(json.dumps(item,indent=2))
