import json,os,signal,subprocess,time
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=r/'src_raw_grounding_v31';out=r/'raw_successor_features_v31'
assert not (out/'parallel_render_launcher.json').exists()
workers=json.loads((out/'parallel_launcher.json').read_text())['workers']
assert len(workers)==1 and workers[0]['gpu']==6
pid=workers[0]['pid'];path=Path('/proc',str(pid),'cmdline')
if path.exists():
    command=path.read_bytes().replace(b'\0',b' ').decode()
    assert str(source/'collect_successor_features.py') in command and str(out) in command
    os.killpg(pid,signal.SIGTERM)
    for _ in range(40):
        if not path.exists():break
        time.sleep(.25)
    assert not path.exists(),'Existing collector did not exit'
used=int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<256,('Other GPU6 work appeared; do not interrupt it',used)
args=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'collect_successor_features_parallel.py'),
    '--out',str(out),'--checkpoint',str(r/'run_planning_observed_v29/checkpoint_002000.pt'),'--workers','4']
with (out/'parallel_render.log').open('xb',buffering=0) as log:
    process=subprocess.Popen(args,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
record=dict(pid=process.pid,command=args,gpu=6,created=time.time(),supersedes=pid,rendering_workers=4,
    old_completed_data_preserved=True,training_process_untouched=True,additional_evaluations=False)
(out/'parallel_render_launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
