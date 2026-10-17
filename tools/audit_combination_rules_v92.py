"""Canonical CUDA arithmetic, archived beliefs only; no navigation evaluation."""
import json, os, shlex
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1]/'output/state_combinations_v92'
code=r'''
import os
os.environ['OMP_NUM_THREADS']='1'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'state_combinations_v92'
sys.path.insert(0,str(b/'parallel_ppo_v85/src'))
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=True
source=json.loads((out/'source.json').read_text());record=source['checkpoint']
p=b/'parallel_ppo_v85/best_development'/record['checkpoint_file']
assert hashlib.sha256(p.read_bytes()).hexdigest()==record['checkpoint_sha256']
saved=torch.load(p,map_location='cpu',weights_only=False)
model=ProbabilisticTokenAutomaton(**saved['manifest']['architecture']).cuda().eval()
model.load_state_dict(saved['model']);model.requires_grad_(False);del saved
measurements=json.loads((b/'state_semantics_v91/measurements.json').read_text())
pg=json.loads((b/'state_semantics_v91/interventions.json').read_text())['programs']
pg={(p['episode_id'],p['start']):p['coefficients'] for p in pg}
seen=set();proofs=[];max_error=0.;mismatch=0
with torch.no_grad():
 for r in measurements['rows']:
  k=(r['category'],r['scene'],r['image_sha256'])
  if k in seen:continue
  seen.add(k);task=['chair','table'].index(r['category'])
  c=torch.tensor(pg[r['episode_id'],r['program_start']],device='cuda').reshape(1,-1).repeat(8,1)
  program=model.program_from_coefficients(c,r['program_start'],torch.tensor([task]*8,device='cuda'))
  belief=torch.tensor(r['belief'],device='cuda')[None].repeat(8,1,1)
  probability,evidence=model.apply_action_rules(program,belief,r['step'])
  expected=torch.tensor(r['probabilities'],device='cuda')
  max_error=max(max_error,float((probability[0]-expected).abs().max()))
  mismatch+=int(probability[0].argmax()!=r['action'])
  tables=(model.conclusions[program.tasks]+program.action_delta).softmax(-1)
  action=r['action'];terms=[]
  for j,(i,k) in enumerate(model.pairs):
   weight=program.rule_weights[0,j]
   contribution=weight*evidence[0,j,action]
   each=weight*belief[0,i,:,None]*belief[0,k,None,:]*tables[0,j,:,:,action]
   idx=int(each.argmax());u,v=divmod(idx,16)
   terms.append(dict(factors=[i,k],weight=float(weight),support=float(contribution),
    selected_action_share=float(contribution/probability[0,action]),top_equality_ids=[u,v],
    premise_probability=float(belief[0,i,u]*belief[0,k,v]),rule_action_probability=float(tables[0,j,u,v,action]),
    top_term_contribution=float(each[u,v]),pair_preferred_action=int(evidence[0,j].argmax())))
  proofs.append(dict(episode_id=r['episode_id'],step=r['step'],category=r['category'],observed=r['observed'],
    memory=r['memory'],action=action,probability=float(probability[0,action]),pairs=terms))
assert max_error<2e-5 and mismatch==0,(max_error,mismatch)
result=dict(checkpoint=record,proofs=proofs,maximum_rule_replay_error=max_error,action_mismatches=mismatch,
 time=time.time(),scope='Canonical batch8 CUDA action-rule replay on frozen saved beliefs. No encoder, simulator, or optimizer.')
(out/'rule_proofs.json').write_text(json.dumps(result))
print(json.dumps({k:v for k,v in result.items() if k!='proofs'}))
'''
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.129.2.34',username='user',password=os.environ['STRIPS_SSH_PASSWORD'],timeout=15,look_for_keys=False,allow_agent=False)
try:
    _,stdout,stderr=client.exec_command('/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -c '+shlex.quote(code),timeout=90)
    message=stdout.read().decode();error=stderr.read().decode()
    assert stdout.channel.recv_exit_status()==0,error
    print(message)
    with client.open_sftp() as sftp:sftp.get('/home/user/xjy/nesy_vla_20261010/state_combinations_v92/rule_proofs.json',str(ROOT/'rule_proofs.json'))
finally:client.close()
