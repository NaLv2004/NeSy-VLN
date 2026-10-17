import json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';out=base/'goal_condition_ablation_v13/task_tokens'
assert not out.exists();out.mkdir()
shutil.copy2(base/'symbolic_reward_v11/configs/original.json',root/'configs/original.json')
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(base/'probe_paired_goal_learning.py'),
    '--root',str(root),'--data',str(base/'paired_goal_probe_v12/input_snapshot'),
    '--checkpoint',str(base/'symbolic_reward_v11/initialization_source'),'--out',str(out),'--steps','160','--task-token-encoder']
with (out/'probe.log').open('ab') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
record={'pid':child.pid,'name':'task_tokens','time':time.time(),'command':cmd}
(out/'probe_process.json').write_text(json.dumps(record,indent=2))
manifest=base/'goal_condition_ablation_v13/processes.json';items=json.loads(manifest.read_text());items.append(record)
manifest.write_text(json.dumps(items,indent=2));print(json.dumps(record,indent=2))
