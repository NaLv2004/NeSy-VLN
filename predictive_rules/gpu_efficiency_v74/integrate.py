"""Produce accelerated sources without changing any live training process."""
import ast,shutil
from pathlib import Path
BASE=Path(__file__).parent.parent;HERE=Path(__file__).parent

def accelerate_eval(path):
    s=path.read_text()
    s=s.replace('from planner import Program,plan,audit','from gpu_planner import Program,plan,audit,extend_batch')
    s=s.replace('requests=[]\n', 'requests=[];prepared=[]\n',1)
    marker="                        proof=plan(programs[i],memories[i],observed,horizon=32-step)"
    if marker not in s:marker="                        if history_predictor is None:proof=plan(programs[i],memories[i],observed,horizon=32-step)"
    assert marker in s
    prefix="""                        prepared.append((i,observed,observed_before_renewal,renew,probability))
                    extend_batch([programs[i] for i in active],32-step,device)
                    for i,observed,observed_before_renewal,renew,probability in prepared:
"""
    s=s.replace(marker,prefix+marker)
    ast.parse(s);path.write_text(s)

# v71 has not begun: its launcher is waiting for the complete1200 collection.
folder=BASE/'expanded_dynamics_v71'
for name in ('gpu_planner.py','autobatch.py'):shutil.copy2(HERE/name,folder/name)
accelerate_eval(folder/'evaluate.py')
s=(folder/'train_joint.py').read_text().replace('import argparse,json,os,time','import argparse,json,os,time,gc')
s=s.replace('from accelerated_transition import install','from accelerated_transition import install\nfrom autobatch import choose')
s=s.replace("    @torch.no_grad()\n    def score", "    if first==1:\n        selected_batch,capacity=choose(model,objective,data,rng)\n        feedback_size=selected_batch['feedback'];pair_size=selected_batch['transitions']\n        if rank==0:atomic(OUT/'capacity.json',capacity)\n\n    @torch.no_grad()\n    def score")
s=s.replace('        if update==5:', '        if False: # Capacity was measured on the complete dataset before training.')
s=s.replace("model.load_state_dict(torch.load(p,map_location=device,weights_only=False)['model']);model.detach_memory_carrier=True;model.eval()",
"model.load_state_dict(torch.load(p,map_location=device,weights_only=False)['model']);model.detach_memory_carrier=True;model.eval()\n        del wrapper,objective,optimizer,data,loss,metrics;gc.collect();torch.cuda.empty_cache()")
s=s.replace('workers=12','workers=24')
s=s.replace("inference='Unchanged NumPy rule-table probability derivation; memory grounded only on rule renewal'",
"inference='Same float64 discrete-rule Bellman derivation batched on GPU; memory grounded only on rule renewal',gpu_symbolic_planner=True,validation_workers_per_gpu=24")
ast.parse(s);(folder/'train_joint.py').write_text(s)
print('Prepared pending v71: measured large batches and batched GPU symbolic evaluation.')
