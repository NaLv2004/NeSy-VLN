import hashlib,json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');source=base/'paired_goal_probe_v12';root=base/'goal_condition_ablation_v13'
assert not root.exists();root.mkdir()
baseline=json.loads((source/'result.json').read_text());assert baseline['state']=='complete'
records=[]
for name,style,weight in [('goal_only','goal_only',0.),('pair_ranking','full',1.)]:
    out=root/name;out.mkdir()
    cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',
        str(base/'probe_paired_goal_learning.py'),'--root',str(base/'symbolic_reward_v11'),
        '--data',str(source/'input_snapshot'),'--checkpoint',str(base/'symbolic_reward_v11/initialization_source'),
        '--out',str(out),'--steps','160','--task-style',style,'--pair-ranking-weight',str(weight)]
    with (out/'probe.log').open('ab') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    item={'name':name,'pid':child.pid,'time':time.time(),'command':cmd};records.append(item)
    (out/'probe_process.json').write_text(json.dumps(item,indent=2))
(root/'processes.json').write_text(json.dumps(records,indent=2))
(root/'experiment.json').write_text(json.dumps({'time':time.time(),'baseline':str(source/'result.json'),
    'changed_one_factor_each':True,'data':str(source/'input_snapshot'),
    'shared_source_sha':baseline['source_commit']['checkpoint_sha256'],
    'no_navigation_performance_claim':True,'all_actions_generated_rule_inference':True,
    'script_sha256':hashlib.sha256((base/'probe_paired_goal_learning.py').read_bytes()).hexdigest()},indent=2))
print(json.dumps(records,indent=2))
