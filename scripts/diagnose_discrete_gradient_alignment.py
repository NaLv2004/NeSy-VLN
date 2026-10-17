"""Compare surrogate derivatives with actual one-bit changes of a saved proof."""
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
logic=TotalBranchLogic(32,17,96);logic.load_state_dict(torch.load(folder/'logic_snapshot.pt',map_location='cpu',weights_only=True))
results={}
for temperature in [1.,4.]:
    rows=[]
    for i,row in enumerate(record['steps']):
        p=programs[row['version']].detach();p.action_heads=p.action_heads.clamp_min(1e-30).pow(1/temperature)
        p.action_heads=p.action_heads/p.action_heads.sum(-1,keepdim=True)
        raw=torch.tensor(row['symbol_codes'],dtype=torch.float32,requires_grad=True)
        result=logic.infer(p,tokenize(raw));a=row['action'];loss=-result['probs'][a].clamp_min(1e-9).log()
        gradient=torch.autograd.grad(loss,raw)[0];deltas=[]
        with torch.no_grad():
            for j in range(len(raw)):
                alternate=raw.detach().clone();alternate[j]=1-alternate[j]
                q=logic.infer(p,tokenize(alternate))['probs'];deltas.append(float(-q[a].clamp_min(1e-9).log()-loss))
        delta=raw.new_tensor(deltas);local=gradient*(1-2*raw.detach());meaningful=delta.abs()>1e-4
        mistaken=meaningful & ((local*delta)<0);missing=meaningful & (local.abs()<=1e-6)
        proposed=int(local.argmin());best=int(delta.argmin())
        rows.append({'step':i,'action':a,'nll':float(loss.detach()),'meaningful_flips':int(meaningful.sum()),
            'wrong_direction':int(mistaken.sum()),'zero_surrogate_nonzero_effect':int(missing.sum()),
            'beneficial_flip_exists':bool(delta.min()< -1e-4),'surrogate_proposed_bit':proposed,
            'surrogate_proposal_actual_delta':float(delta[proposed]),'best_bit':best,'best_actual_delta':float(delta[best]),
            'local_linear_deltas':local.tolist(),'actual_loss_deltas':deltas})
    count=sum(r['meaningful_flips'] for r in rows);wrong=sum(r['wrong_direction'] for r in rows)
    results[str(temperature)]={'actions':len(rows),'meaningful_flips':count,'wrong_sign_flips':wrong,
        'wrong_sign_fraction':wrong/max(1,count),'zero_surrogate_nonzero_effect':sum(r['zero_surrogate_nonzero_effect'] for r in rows),
        'surrogate_proposal_increases_loss':sum(r['surrogate_proposal_actual_delta']>1e-4 for r in rows),
        'beneficial_flip_exists':sum(r['beneficial_flip_exists'] for r in rows),'rows':rows}
out={'time':time.time(),'source_trace':str(folder),'episode':record['episode_id'],'results':results,
     'scope':'Exact counterfactual symbolic proof, no image inputs, no environment actions, no training or fixed-validation updates.'}
(root/'discrete_gradient_alignment.json').write_text(json.dumps(out,indent=2))
print(json.dumps({name:{k:v for k,v in value.items() if k!='rows'} for name,value in results.items()},indent=2))
