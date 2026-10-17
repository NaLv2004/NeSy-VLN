import collections,hashlib,json,os,random,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_diverse_history_v70';OUT=ROOT/'diverse_history_v70'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(ROOT/'run_history_transition_v69/supervisor.json')['state']=='finished'
old=read(ROOT/'run_history_transition_v69/supervisor.json')
for p in [old['pid'],*[read(ROOT/f'run_history_transition_v69/status_rank{r}.json')['pid'] for r in (0,1)]]:
    proc=Path(f'/proc/{p}/cmdline');assert not(proc.exists() and proc.read_bytes())
assert not (OUT/'launch.json').exists(),'Existing collection must be inspected instead of duplicated'
for gpu in (6,7):
    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert used<500,(gpu,used)
protocol=read(ROOT/'normal_near_medium_v67/manifest.json');tasks=read(ROOT/'normal_near_medium_v67/train_tasks.json')
assert sha(ROOT/'normal_near_medium_v67/train_tasks.json')==protocol['train_sha256']
assert sha(ROOT/'normal_near_medium_v67/development80.json')==protocol['development_sha256']
excluded=set(protocol['selected_boundary_episodes']);groups=collections.defaultdict(list)
for e in tasks:
    if e['episode_id'] not in excluded:groups[(e['scene'],e['category'],e['distance_group'])].append(e)
rng=random.Random(701017);selected=[];holdout=[]
for key,values in sorted(groups.items()):
    sample=rng.sample(sorted(values,key=lambda e:e['episode_id']),20);selected.extend(sample);holdout.extend(e['episode_id'] for e in sample[:4])
rng.shuffle(selected);assert len(selected)==1200 and len(holdout)==240
assert not set(e['episode_id'] for e in selected)&excluded
OUT.mkdir(exist_ok=True);(OUT/'episodes').mkdir(exist_ok=True)
commit=read(ROOT/'run_goal_probability_v68/best.COMMITTED.json');ckpt=ROOT/'run_goal_probability_v68'/commit['filename'];assert sha(ckpt)==commit['sha256']
shutil.copy2(ckpt,OUT/'source.pt');(OUT/'source.COMMITTED.json').write_text(json.dumps(dict(commit,source=str(ckpt)),indent=2))
manifest=dict(time=time.time(),source=commit,source_checkpoint=str(ckpt),seed=701017,tasks=1200,fit_episodes=960,holdout_episodes=240,
    holdout_ids=sorted(holdout),episodes=selected,excluded_old200=sorted(excluded),curriculum_sha256=protocol['train_sha256'],
    development80_sha256=protocol['development_sha256'],train_layouts=protocol['train_layouts'],
    episode_counts={str(k):sum((e['scene'],e['category'],e['distance_group'])==k for e in selected) for k in sorted(groups)},
    history_policy='Frozen v68selected300. Every executed action comes from the unchanged symbolic table planner.',
    training_probes='All17firstactions plus1nonterminalsecondaction pernonSTOPfirst, every4thposition and atSTOP/no-proof',
    privileged_inputs_to_policy=False,stores_maps=False,stores_coordinates=False,stores_distances=False,stores_collision_labels=False,
    outcome_labels='Existing binary STOP success eligibility only, training data only; symbolic IDs remain learned',
    gpus=[6,7],simulators_per_gpu=24,feature_batch=512,max_actions=32,finalfixed80_used=False)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copytree(ROOT/'src_goal_probability_v68/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
with (OUT/'launcher.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'coordinator.py')],cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=p.pid,source=str(SOURCE),run=str(OUT),gpus=[6,7],tasks=1200,fit=960,holdout=240)
(OUT/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
