"""Capacity test of fixed learned programs, using free discrete facts, not images.

The optimized facts are diagnostic variables only and never used for navigation.
"""
import json,sys,time
from pathlib import Path
import torch
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_reward_v11';sys.path.insert(0,str(root))
from nesy_vla.runtime import setup,read,atomic
from nesy_vla.small_generator import SymbolicProgramGenerator
from nesy_vla.neural_logic import NeuralSymbolicLogic
from nesy_vla.symbols import tokenize
setup(read(root/'configs/original.json'))
from stripsnav.io import load_committed
torch.set_num_threads(2);torch.manual_seed(1011)
state=load_committed(root/'initialization_source');cfg=state['config'];spec=cfg['program']
generator=SymbolicProgramGenerator(spec,cfg.get('generator_hidden',64)).eval().requires_grad_(False)
logic=NeuralSymbolicLogic(spec['k'],spec['a'],spec['r']).eval().requires_grad_(False)
for name,module in [('generator',generator),('logic',logic)]:
    module.load_state_dict({k.removeprefix(name+'.'):v for k,v in state['model'].items() if k.startswith(name+'.')})
del state
out=base/'paired_goal_probe_v12/rule_capacity.json';start=time.time();result={}
for category in ['chair','table']:
    with torch.no_grad():program=generator('Find the '+category+' and stop close to it.',[],[]).detach()
    labels=torch.arange(spec['a']).repeat(3);logits=torch.nn.Parameter(torch.randn(len(labels),spec['k']))
    optimizer=torch.optim.Adam([logits],lr=.12);best=[{'probability':-1.} for _ in range(spec['a'])]
    for update in range(81):
        probabilities=[];states=tokenize(logits.sigmoid())
        for i,target in enumerate(labels.tolist()):
            # A separate validated discrete state per action/restart.
            code=tokenize(logits[i].sigmoid());proof=logic.infer(program,code);p=proof['probs']
            probabilities.append(p[target])
            if float(p[target].detach())>best[target]['probability']:
                best[target]={'probability':float(p[target].detach()),'top_action':int(p.detach().argmax()),
                    'symbols':code.codes.tolist(),'update':update}
        if update<80:
            loss=-torch.stack(probabilities).clamp_min(1e-9).log().mean();optimizer.zero_grad();loss.backward();optimizer.step()
    result[category]={'reachable_actions':sum(i==r['top_action'] for i,r in enumerate(best)),
        'actions':spec['a'],'best':best}
    atomic(out,{'state':'running','time':time.time(),'results':result,'not_navigation':True})
    print(json.dumps({'category':category,'reachable':result[category]['reachable_actions']}),flush=True)
atomic(out,{'state':'complete','time':time.time(),'results':result,'seconds':time.time()-start,
    'source':str(root/'initialization_source'),'not_navigation':True,
    'qualification':'Found Boolean states only demonstrate rule capacity, not visually grounded or reachable physical states.'})
