import json,time,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'counterfactual_proof_v11'
def read(p):return json.loads(p.read_text()) if p.exists() else None
launch=read(out/'launchers.json');records=[]
for item in launch['processes']:
    proc=Path('/proc')/str(item['pid']);alive=proc.exists()
    row=dict(**item,alive=alive,status=read(out/f"shard_{item['shard']}"/'status.json'),complete=read(out/f"shard_{item['shard']}"/'COMPLETE.json'))
    if alive:
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        row['identity_verified']=str(root/'src_proof_collection_v11/collect_proof_transitions.py') in command and str((proc/'cwd').resolve())==str(root/'src_proof_collection_v11')
    log=out/f"driver_{item['shard']}.log";row['log_tail']=log.read_text(errors='replace')[-900:] if log.exists() else None
    records.append(row)
print(json.dumps(dict(time=time.time(),collection=records,files=len(list(out.glob('*.npz'))),
    no_proof_skips=len(list((out/'proofs').glob('*/NO_PROOF.json'))),
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
