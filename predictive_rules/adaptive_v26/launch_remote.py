"""Launch only the isolated adaptive-rule experiment on the free GPU7."""
import json,os,subprocess,time
from pathlib import Path

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_adaptive_observed_v26';out=root/'run_adaptive_observed_v26'
launcher=root/'adaptive_observed_v26_launcher.json'
assert not out.exists() and not launcher.exists()
used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>256:raise RuntimeError(f'GPU7 is occupied: {used} MiB; no process will be killed')
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
command=[python,'-u',str(source/'train.py'),'--source',str(root/'run_state512_v25/checkpoint_003000.pt'),
         '--out',str(out),'--batch-size','64','--updates','2000','--eval-every','500']
with (root/'adaptive_observed_v26.log').open('xb',buffering=0) as log:
    process=subprocess.Popen(command,cwd=source,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,command=command,gpu=7,created=time.time(),out=str(out),
            source=str(source),goal_complete=False)
launcher.write_text(json.dumps(record,indent=2));print(json.dumps(record))
