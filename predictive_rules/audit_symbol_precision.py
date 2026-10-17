"""Compare AMP training symbols to FP32 inference symbols on training data."""
import json,os,random,subprocess
from pathlib import Path
import torch
import numpy as np
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config,export_program
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.graph_subgoal_interpreter import plan
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<4000, 'Diagnostic requires memory headroom; do not stop other jobs'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
saved=torch.load(root/'run_grounded_rules_v9/checkpoint_001200.pt',map_location='cpu',weights_only=False)
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).cuda().eval();m.load_state_dict(saved['model'])
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
rows=[]
with torch.no_grad():
    for start in range(0,128,16):
        data=batch(selected[start:start+16],'cuda');results=[]
        for amp in [False,True]:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                pg=m.generate(data['task'],data['history'][:,:1],data['previous_actions'][:,:1],data['history_length'],start=0)
                q=m.training_targets(pg,data['targets'],data['actions']).squeeze(2)
            programs=[export_program(pg,i) for i in range(16)]
            results.append((q.cpu(),programs))
        for i,w in enumerate(selected[start:start+16]):
            n=int(w['goal_index']);fp32,amp=results[0][0][i,:n+1].argmax(-1).numpy(),results[1][0][i,:n+1].argmax(-1).numpy()
            p0,p1=results[0][1][i],results[1][1][i]
            route0=plan(p0,np.array([fp32[0]]));route1=plan(p1,np.array([amp[0]]))
            rows.append(dict(episode=w['episode_id'],states=n+1,changed_states=int((fp32!=amp).sum()),
                initial_changed=bool(fp32[0]!=amp[0]),goal_changed=bool(p0.subgoal[0]!=p1.subgoal[0]),
                changed_rules=int((p0.transitions.argmax(-1)!=p1.transitions.argmax(-1)).sum()),
                rules=int(np.prod(p0.transitions.shape[:2])),plans_equal=route0['actions']==route1['actions'],
                first_equal=route0['actions'][:1]==route1['actions'][:1],fp32=fp32.tolist(),amp=amp.tolist()))
result=dict(training_only=True,checkpoint='grounded_rules_v9_step1200',samples=len(rows),
    **{k:sum(r[k] for r in rows) for k in ['states','changed_states','initial_changed','goal_changed','changed_rules','rules','plans_equal','first_equal']},
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,rows=rows)
(root/'diagnostics/symbol_precision_v9.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
