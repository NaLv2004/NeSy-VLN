import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');code=base/'diagnostics_20261011'
out=base/'symbolic_sft_v5/diagnostics/plain30_transport_development';out.mkdir(parents=True,exist_ok=True)
assert not (out/'process.json').exists(),'Inspect the existing process; do not duplicate the experiment'
free=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=3000,'Insufficient headroom for the small independent diagnostic'
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
command=[python,'-u',str(code/'evaluate_transport_alignment.py'),'--run',str(base/'symbolic_sft_v5/runs/sft_plain'),
    '--step','30','--output',str(out)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
with (out/'process.log').open('ab') as log:
    process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'time':time.time(),'command':command,'gpu':6,'available_mib_before':free}
(out/'process.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
