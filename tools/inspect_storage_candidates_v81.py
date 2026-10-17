import json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');r=Path('/home/user/xjy/strips_wm_navigation_20261009/runs')
config=json.loads((b/'storage_reward_inspection_v81.json').read_text())['current_reward_config']
env=json.loads((Path(config['reference_root'])/'configs/experiment.json').read_text())
large=[]
for p in r.rglob('*'):
 if p.is_file() and not p.is_symlink() and p.stat().st_size>150*2**20:
  st=p.stat();large.append({'path':str(p),'gib':round(st.st_size/2**30,3),'inode':st.st_ino,'links':st.st_nlink,'mtime':st.st_mtime})
evidence=[]
for p in (b/'evidence_archives/content_store').rglob('*'):
 if p.is_file() and p.stat().st_size>100*2**20:evidence.append({'path':str(p),'gib':round(p.stat().st_size/2**30,3)})
result={'time':time.time(),'large_checkpoint_files':sorted(large,key=lambda x:-x['gib']),
 'large_evidence':sorted(evidence,key=lambda x:-x['gib']),
 'environment':env,'nproc':os.cpu_count(),'meminfo':Path('/proc/meminfo').read_text().splitlines()[:5],
 'mounts':subprocess.check_output(['df','-h','-T'],text=True),
 'control_processes':json.loads((b/'perceptual_rl_v80/action_rules_only/processes.json').read_text())}
(b/'storage_candidates_v81.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
