"""Inspect goal-conditioned learned programs; no images or held-out rollouts."""
import json,sys,time,gc
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_joint_v9';sys.path.insert(0,str(root))
import torch
from nesy_vla.runtime import read,setup,atomic
from nesy_vla.small_generator import SymbolicProgramGenerator
cfg=read(root/'configs/joint.json');setup(cfg)
from stripsnav.io import load_committed
torch.set_num_threads(2);results=[]
paths=[base/'symbolic_robust_v7/runs/robust_reference/evaluations/step_000040/checkpoint',
    root/'initialization_source',base/'symbolic_vision_v8/runs/frozen_qwen/evaluations/step_000010/checkpoint']
for path in paths:
    state=load_committed(path);config=state['config'];model=SymbolicProgramGenerator(config['program'],config.get('generator_hidden',64)).eval()
    model.load_state_dict({k.removeprefix('generator.'):v for k,v in state['model'].items() if k.startswith('generator.')});del state;gc.collect()
    with torch.no_grad():
        tasks=['Find the '+x+' and stop close to it.' for x in ['chair','table']]
        contexts=[model.context(task,[],[]) for task in tasks];p,q=[model(task,[],[]) for task in tasks]
        fields={}
        for name in ['definitions','premises','state_heads','action_heads','effects','action_conditions','weights']:
            a,b=getattr(p,name),getattr(q,name);fields[name]={'rms_difference':float((a-b).square().mean().sqrt()),'max_difference':float((a-b).abs().max()),
                'argmax_differences':int((a.argmax(-1)!=b.argmax(-1)).sum()) if a.ndim>1 else None}
        results.append({'checkpoint':str(path),'commit':read(path/'COMMITTED'),'context_rms_difference':float((contexts[0]-contexts[1]).square().mean().sqrt()),'program':fields})
        print(json.dumps(results[-1]),flush=True)
    del model;gc.collect()
atomic(root/'goal_conditioning_audit.json',{'time':time.time(),'results':results,'qualification':'Measures generated initial program dependence on the target word; not a navigation success evaluation.'})
