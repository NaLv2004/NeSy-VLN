"""Recompute the exact symbolic proof on immutable saved first-update traces.

No RGB, poses, distances or target semantics enter this diagnosis.
"""
import json,sys,time
from pathlib import Path
import torch
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_branch_v17');sys.path.insert(0,str(root))
from nesy_vla.program import Program
from nesy_vla.branch_logic import TotalBranchLogic
from nesy_vla.symbols import tokenize
torch.set_num_threads(2)
folder=root/'runs/branch_current/demonstration_traces/update_000001'
record=json.loads((folder/'trajectory.json').read_text());programs={}
for file in folder.glob('program_*.json'):
    p=Program.from_record(json.loads(file.read_text())['program']);programs[p.version]=p
logic=TotalBranchLogic(32,17,96)
logic.load_state_dict(torch.load(folder/'logic_snapshot.pt',map_location='cpu',weights_only=True))
results={}
for temperature in [1.,2.,4.,8.]:
    rows=[]
    for i,row in enumerate(record['steps']):
        p=programs[row['version']].detach();p.action_heads=p.action_heads.clamp_min(1e-30).pow(1/temperature)
        p.action_heads=p.action_heads/p.action_heads.sum(-1,keepdim=True)
        raw=torch.tensor(row['symbol_codes'],dtype=torch.float32,requires_grad=True)
        result=logic.infer(p,tokenize(raw));action=torch.tensor(row['action'])
        loss=-torch.distributions.Categorical(probs=result['probs']).log_prob(action)
        gradient=torch.autograd.grad(loss,raw)[0]
        saved=torch.tensor(row['support']);saved=saved/saved.sum()
        rows.append({'step':i,'action':row['action'],'predicted':int(result['probs'].argmax()),
            'target_probability':float(result['probs'][action].detach()),'nll':float(loss.detach()),
            'symbol_gradient_norm':float(gradient.norm()),'max_symbol_gradient':float(gradient.abs().max()),
            'replay_max_error':float((result['probs'].detach()-saved).abs().max()),
            'positive_branches':int((result['action_activation']>0).sum()),
            'max_branch_mass':float(result['action_activation'].max().detach()),
            'max_leaf_action_probability':float(p.action_heads[logic.action_slice].max()),
            'coverage_mass':float(result['coverage_mass'].detach())})
    results[str(temperature)]={'rows':rows,'mean_nll':sum(r['nll'] for r in rows)/len(rows),
        'max_symbol_gradient_norm':max(r['symbol_gradient_norm'] for r in rows),
        'mean_symbol_gradient_norm':sum(r['symbol_gradient_norm'] for r in rows)/len(rows),
        'max_replay_error':max(r['replay_max_error'] for r in rows),
        'correct_actions':sum(r['predicted']==r['action'] for r in rows),'actions':len(rows)}
result={'time':time.time(),'source_trace':str(folder),'episode':record['episode_id'],
        'only_saved_discrete_states_and_generated_rules':True,'navigation_evaluation':False,'results':results}
out=root/'initial_rule_gradient_diagnosis.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='results'}))
print(json.dumps({t:{k:v for k,v in r.items() if k!='rows'} for t,r in results.items()},indent=2))
