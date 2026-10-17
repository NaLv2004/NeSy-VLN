"""One new experiment, only after v67 is authoritatively finished."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_goal_probability_v68';OUT=ROOT/'run_goal_probability_v68';DATA=ROOT/'goal_probability_v68'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
previous=ROOT/'run_normal_near_medium_v67';result=read(previous/'result.json');supervisor=read(previous/'supervisor.json')
assert result['state']=='complete' and supervisor['state']=='finished'
for p in [supervisor['pid'],*[read(previous/f'status_rank{i}.json')['pid'] for i in (0,1)]]:
    proc=Path(f'/proc/{p}/cmdline');assert not (proc.exists() and proc.read_bytes()),('Still-live prior process',p)
assert not (OUT/'training_launch.json').exists(),'Existing experiment must be inspected rather than restarted'
for gpu in (6,7):
    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert used<500,(gpu,used)
baseline=read(previous/'baseline_v63_dev80/summary.json')
if result.get('development') and result['development']['successes']>baseline['successes']:
    commit=result['chosen'];path=previous/commit['filename'];reason='v67 improved full unchanged current80'
else:
    commit=read(ROOT/'goal_discrimination_v64/source.COMMITTED.json');path=ROOT/'goal_discrimination_v64/source.pt';reason='v67 did not improve current80 or failed its predefined holdout screen; retain proven source'
assert sha(path)==commit['sha256']
DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
shutil.copy2(path,DATA/'source.pt');(DATA/'source.COMMITTED.json').write_text(json.dumps(dict(commit,source_path=str(path)),indent=2))
selection=dict(time=time.time(),source=str(path),source_sha256=commit['sha256'],reason=reason,
    current80_baseline=baseline['successes'],previous_development=result.get('development'),
    curriculum_hash=read(ROOT/'normal_near_medium_v67/manifest.json')['development_sha256'],
    finalfixed80_used=False,hand_defined_state_ids=False,learned_reward_or_value_head=False,
    objective='v67 joint discrete evidence/dynamics plus measured fixed-sequence symbolic goal probability calibration')
(DATA/'source_selection.json').write_text(json.dumps(selection,indent=2))
shutil.copytree(ROOT/'src_normal_near_medium_v67/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
with (OUT/'launcher.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=p.pid,source=str(SOURCE),run=str(OUT),gpus=[6,7],selection=selection)
(OUT/'training_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
