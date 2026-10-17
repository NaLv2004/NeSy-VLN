import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'execution_development_v36';out.mkdir(exist_ok=True)
script=base/'symbolic_paired_v12/scripts/evaluate_execution_probes_v36.py'
assert not (out/'launcher.json').exists(),'Inspect existing launch instead of duplicating it'
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    if proc.parent.name==str(os.getpid()):continue
    try:assert str(script).encode() not in proc.read_bytes(),'Evaluation already live'
    except OSError:pass
with (out/'driver.log').open('ab') as log:
    process=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script)],
        cwd=script.parents[1],stdout=log,stderr=log,start_new_session=True)
record={'pid':process.pid,'time':time.time(),'command':str(script),'automatic_restart':False,
    'original_training_not_interrupted':True,'evaluation':'Original80development tasks for two new distinct model weights; no fixed80'}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
