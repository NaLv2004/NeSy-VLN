import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12'
study=base/'goal_condition_capacity_v14';study.mkdir(exist_ok=True)
specs=[('task_tokens_800',[]),('task_tokens_visual_800',['--visual-source',
    '/home/user/xjy/strips_wm_navigation_20261009/runs/search_cost_ppo_belief_v2_step000200/best'])]
records=[]
for name,flags in specs:
    out=study/name;assert not out.exists();out.mkdir()
    cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(base/'probe_paired_goal_learning.py'),
        '--root',str(root),'--data',str(base/'paired_goal_probe_v12/input_snapshot'),
        '--checkpoint',str(base/'symbolic_reward_v11/initialization_source'),'--out',str(out),
        '--steps','800','--task-token-encoder',*flags]
    with (out/'probe.log').open('ab') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    record={'name':name,'pid':child.pid,'command':cmd,'time':time.time()};records.append(record)
    (out/'process.json').write_text(json.dumps(record,indent=2))
(study/'processes.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
