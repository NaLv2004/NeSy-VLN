"""Read a frozen archived actor's rule tensors; never run or modify training."""
import hashlib, json, os, shlex
from pathlib import Path
import paramiko

ROOT = Path(__file__).resolve().parents[1] / 'output/state_combinations_v92'
ROOT.mkdir(exist_ok=True)
REMOTE = '/home/user/xjy/nesy_vla_20261010/state_combinations_v92'
code = r'''
import hashlib,json
from pathlib import Path
import numpy as np,torch
torch.set_num_threads(1)
b=Path('/home/user/xjy/nesy_vla_20261010')
out=b/'state_combinations_v92';out.mkdir(exist_ok=True)
record=json.loads((b/'state_semantics_v91/interventions.json').read_text())['checkpoint']
p=b/'parallel_ppo_v85/best_development'/record['checkpoint_file']
assert hashlib.sha256(p.read_bytes()).hexdigest()==record['checkpoint_sha256']
c=torch.load(p,map_location='cpu',weights_only=False)
keys=['conclusions','conclusion_basis','weight_logits','weight_basis']
np.savez_compressed(out/'action_rules.npz',**{k:c['model'][k].detach().numpy() for k in keys})
(out/'source.json').write_text(json.dumps({'checkpoint':record,'architecture':c['manifest']['architecture'],
 'scope':'Read-only CPU export of symbolic action rules. No model inference, environment action, or optimization.',
 'sha256':hashlib.sha256((out/'action_rules.npz').read_bytes()).hexdigest()}))
print(json.dumps({'state':'complete','bytes':(out/'action_rules.npz').stat().st_size}))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.129.2.34', username='user', password=os.environ['STRIPS_SSH_PASSWORD'],
               timeout=15, look_for_keys=False, allow_agent=False)
try:
    command='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -c '+shlex.quote(code)
    _, stdout, stderr = client.exec_command(command, timeout=60)
    message=stdout.read().decode(); error=stderr.read().decode()
    assert stdout.channel.recv_exit_status()==0,error
    print(message)
    with client.open_sftp() as sftp:
        for name in ['action_rules.npz','source.json']:
            sftp.get(REMOTE+'/'+name, str(ROOT/name))
finally:
    client.close()
record=json.loads((ROOT/'source.json').read_text())
assert hashlib.sha256((ROOT/'action_rules.npz').read_bytes()).hexdigest()==record['sha256']
