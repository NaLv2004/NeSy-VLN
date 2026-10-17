"""Remote-only freeze of the first 768 preselected tasks and isolated launches."""
from pathlib import Path
import collections,hashlib,json,os,shutil,subprocess,time
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
original=ROOT/'diverse_history_v70';data=ROOT/'parallel_prefix_v75'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(original/'manifest.json');episodes=manifest['episodes'][:768]
assert len(episodes)==768
# The prefix is by the PRE-EXPERIMENT manifest order, never completion order or success.
for e in episodes:
    p=original/'episodes'/(e['episode_id']+'.npz')
    assert p.exists() and p.with_suffix('.json').exists(),('Prefix not committed',e['episode_id'])
names={e['episode_id'] for e in episodes};held=set(manifest['holdout_ids'])&names
assert held and names-held
if not data.exists():
    data.mkdir();(data/'episodes').mkdir()
    for e in episodes:
        for suffix in ('.npz','.json'):
            name=e['episode_id']+suffix;(data/'episodes'/name).symlink_to(original/'episodes'/name)
    for name in ('source.pt','source.COMMITTED.json'):(data/name).symlink_to(original/name)
    frozen=dict(manifest,episodes=episodes,holdout_ids=sorted(held),tasks=768,fit_episodes=768-len(held),holdout_episodes=len(held),
        selection='First768 entries of the original pre-outcome randomized1200manifest; no selection on success or collection speed',
        parent_manifest_sha256=sha(original/'manifest.json'))
    (data/'manifest.json').write_text(json.dumps(frozen,indent=2))
    (data/'status.json').write_text(json.dumps(dict(state='complete',completed=768,time=time.time())))
assert [e['episode_id'] for e in read(data/'manifest.json')['episodes']]==[e['episode_id'] for e in episodes]
records=[]
for name,gpus in [('stable_rules_v75',[2,3]),('expanded_history_v76',[4,5])]:
    source=ROOT/('src_'+name);out=ROOT/('run_'+name);out.mkdir(exist_ok=True)
    if (out/'training_launch.json').exists():
        records.append(dict(existing=True,**read(out/'training_launch.json')));continue
    for gpu in gpus:
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        assert used<500,(gpu,used)
    shutil.copytree(ROOT/'src_goal_probability_v68/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    files={p.name:sha(p) for p in source.glob('*.py')};(out/'source_files.json').write_text(json.dumps(files,indent=2))
    with (out/'launcher.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(['python3','-u',str(source/'supervisor.py')],cwd=source,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row=dict(pid=p.pid,time=time.time(),source=str(source),run=str(out),gpus=gpus,dataset=str(data),
        parent_source_checkpoint=str(original/'source.pt'),source_sha256=read(original/'source.COMMITTED.json')['sha256'])
    (out/'training_launch.json').write_text(json.dumps(row,indent=2));records.append(row)
print(json.dumps(dict(launched=records,prefix_episodes=768,fit=768-len(held),holdout=len(held),
    categories=dict(collections.Counter(e['category'] for e in episodes)),distances=dict(collections.Counter(e['distance_group'] for e in episodes)))))
