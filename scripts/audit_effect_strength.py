"""Explain saved symbolic transitions without reprocessing any image."""
import json,os,sys
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_robust_v7');sys.path.insert(0,str(root))
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from nesy_vla.program import Program
from nesy_vla.neural_logic import NeuralSymbolicLogic
from nesy_vla.symbols import tokenize
from nesy_vla.logic import noisy_or
from nesy_vla.runtime import read,atomic

torch.set_num_threads(2);records=[]
for step in [1,5,10]:
    folder=root/f'runs/robust_multieffect/demonstration_traces/update_{step:06d}'
    if not (folder/'trajectory.json').exists():continue
    trace=read(folder/'trajectory.json');programs={}
    for f in folder.glob('program_*.json'):
        p=Program.from_record(read(f)['program']);programs[p.version]=p
    p=next(iter(programs.values()));k=len(p.definitions);a=p.action_heads.shape[-1]
    logic=NeuralSymbolicLogic(k,a,len(p.weights));logic.load_state_dict(torch.load(folder/'logic_snapshot.pt',map_location='cpu',weights_only=True));logic.eval()
    values=[]
    with torch.no_grad():
        for row in trace['steps']:
            state=tokenize(torch.tensor(row['belief']));p=programs[row['version']];sl=logic.effect_slice
            condition=p.action_conditions[sl,row['action']]
            firing=logic.fire_discrete(p,state.values,torch.ones(k),sl,state)
            effects=noisy_or(firing[:,None]*condition[:,None]*p.effects[sl],0)
            add,delete=effects.chunk(2,-1);prediction=torch.tensor(row['predicted_next_truth'])
            old=(prediction/(1-delete).clamp_min(1e-6)-add)/(1-add).clamp_min(1e-6)
            retention=((old-.5)/(state.values-.5)).clamp(0,1)
            replay,_,_=logic.transition(p,state.values,torch.ones(k),row['action'],retention)
            error=float((replay-prediction).abs().max());assert error<1e-5
            target=torch.tensor(row['observed_next_truth'])>=.5;changed=target!=state.codes.bool()
            # Diagnostic only: remove the uniform 1/A attenuation of action matching.
            q=p.map(lambda x:x);q.action_conditions=(p.action_conditions*a).clamp_max(1.)
            stronger,_,_=logic.transition(q,state.values,torch.ones(k),row['action'],retention)
            values.append({'action':row['action'],'mean_action_condition':float(condition.mean()),
                'max_action_condition':float(condition.max()),'mean_rule_firing_before_action':float(firing.mean()),
                'mean_add':float(add.mean()),'mean_delete':float(delete.mean()),'mean_retention':float(retention.mean()),
                'changed_bits':int(changed.sum()),'changed_correct':int(((prediction>=.5)==target)[changed].sum()),
                'diagnostic_changed_correct':int(((stronger>=.5)==target)[changed].sum()),
                'stable_bits':int((~changed).sum()),'stable_correct':int(((prediction>=.5)==target)[~changed].sum()),
                'diagnostic_stable_correct':int(((stronger>=.5)==target)[~changed].sum()),'replay_error':error})
    totals={name:sum(x[name] for x in values) for name in ['changed_bits','changed_correct','diagnostic_changed_correct','stable_bits','stable_correct','diagnostic_stable_correct']}
    means={name:sum(x[name] for x in values)/len(values) for name in values[0] if name.startswith('mean_')}
    records.append({'update':step,'episode_id':trace['episode_id'],'frames':len(values),'totals':totals,'means':means,'rows':values})
result={'records':records,'qualification':'Exact symbolic replay and untrained action-matching counterfactual on training traces, not a deployed policy or navigation success.'}
atomic(root/'effect_strength_audit.json',result);print(json.dumps({'records':[{k:v for k,v in r.items() if k!='rows'} for r in records]},indent=2))
