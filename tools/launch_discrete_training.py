"""Launch two independent program generators on free GPUs, keeping frozen splits."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/discrete_v4')
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={int(a):int(b) for a,b in (line.split(',') for line in usage.strip().splitlines())}
jobs=[]
for name,gpu in [('discrete_gru',6),('discrete_vlm',7)]:
    if used[gpu]>1500:raise RuntimeError(f'GPU {gpu} occupied: {used[gpu]} MiB')
    smoke=root/'runs'/f'{name}_smoke'
    if json.loads((smoke/'status.json').read_text()).get('state')!='complete':raise RuntimeError('Startup check incomplete')
    out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'supervisor.json').exists():raise RuntimeError('Run already initialized; inspect rather than duplicate')
    cfg=json.loads((root/'configs'/f'{name}.json').read_text())
    assert cfg['horizon']==32 and cfg['revision_interval']==10 and cfg['discrete_reasoning']
    manifest={'time':time.time(),'initialization':'fresh pretrained image encoder; Qwen base only for VLM variant',
        'generator':cfg['generator_backend'],'discrete_symbols':32,'automaton_memory':'quantized generated effects, reconciled with discrete observation tokens',
        'no_visual_action_branch':True,'predicate_semantics':'automatically learned without semantic labels',
        'split':'existing split_lock and immutable fixed80','startup_checks':'31 base tests plus symbolic memory update test',
        'comparison_limit':'independent learned initialization; success must be measured in closed-loop evaluation'}
    (out/'lineage.json').write_text(json.dumps(manifest,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
    command=[python,'-u','scripts/supervise.py','--out',str(out),'--config',f'configs/{name}.json','--gpus',str(gpu)]
    with (out/'supervisor.log').open('ab') as log:
        p=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append({'supervisor_pid':p.pid,'gpu':gpu,'run':str(out)})
print(json.dumps(jobs,indent=2))
