from pathlib import Path
import ast,shutil
BASE=Path(__file__).parent.parent;HERE=Path(__file__).parent
def write(p,s):ast.parse(s);p.write_text(s)

names=['stable_rules_v75','expanded_history_v76']
for name,parent in zip(names,['expanded_dynamics_v71','compact_history_v73']):
    folder=BASE/name;folder.mkdir(exist_ok=True)
    for p in (BASE/parent).glob('*.py'):
        if p.name not in ('deploy.py','launch_handoff.py','handoff.py','collect_completion.py','record_results.py','audit_aliasing.py','read_status.py'):
            shutil.copy2(p,folder/p.name)
    for p in ('gpu_planner.py','autobatch.py','history_autobatch.py'):shutil.copy2(HERE/p,folder/p)
    d=(BASE/'expanded_dynamics_v71/joint_data.py').read_text().replace("old=ROOT/'diverse_history_v70'","old=ROOT/'parallel_prefix_v75'")
    d=d.replace('assert len(names)==1200','assert len(names)==768').replace('assert len(self.holdout)==240','assert 0<len(self.holdout)<len(names)')
    d=d.replace('episodes=1200,fit_episodes=960,holdout_episodes=240','episodes=len(names),fit_episodes=len(names)-len(self.holdout),holdout_episodes=len(self.holdout)')
    write(folder/'joint_data.py',d)

f=BASE/names[0]
s=(f/'train_joint.py').read_text().replace('expanded_dynamics_v71',names[0]).replace("=='6,7'","=='2,3'").replace("old=ROOT/'diverse_history_v70'","old=ROOT/'parallel_prefix_v75'")
s=s.replace('on1200fresh','on768fresh').replace('960fitting and240heldout','pre-frozen fitting/heldout')
s=s.replace("frozen_context_definitions_goal_and_memory_predictor=True,","frozen_context_definitions_goal_and_memory_predictor=True,visual_evidence_frozen=True,")
s=s.replace('only generic visual residual and observation-transition decoder train','only observation-transition decoder trains; source44 visual and goal symbols remain fixed')
write(f/'train_joint.py',s)
s=(f/'base_objective.py').read_text().replace("TRAINABLE=('linear_visual_evidence.','definition_rule.')","TRAINABLE=('definition_rule.',)")
write(f/'base_objective.py',s)
s=(f/'autobatch.py').read_text().replace('((16384,2048),(16384,4096),(32768,5120),(32768,6144))','((16384,2048),(16384,4096),(32768,6144),(32768,8192),(32768,12288))')
write(f/'autobatch.py',s)
s=(BASE/'continued_evidence_v72/supervisor.py').read_text().replace('continued_evidence_v72',names[0]);write(f/'supervisor.py',s)

f=BASE/names[1]
s=(f/'train_history.py').read_text().replace('compact_history_v73',names[1]).replace('import json,os,time','import json,os,time,gc')
s=s.replace('from accelerated_transition import install','from accelerated_transition import install\nfrom history_autobatch import choose')
s=s.replace('    wrapper=DDP(predictor,device_ids=[local],broadcast_buffers=False)\n','')
s=s.replace('    baseline=score(0)',"    predictor.train();batch,capacity=choose(predictor,data,rng)\n    if rank==0:atomic(OUT/'capacity.json',capacity)\n    wrapper=DDP(predictor,device_ids=[local],broadcast_buffers=False)\n    baseline=score(0)")
s=s.replace('chosen=None;batch=4096;began=time.time()','chosen=None;began=time.time()')
s=s.replace('        if update==5:','        if False: # Full-data capacity was measured before any update.')
s=s.replace("predictor.load_state_dict(torch.load(p,map_location=device,weights_only=False)['predictor']);predictor.eval()",
"predictor.load_state_dict(torch.load(p,map_location=device,weights_only=False)['predictor']);predictor.eval()\n        del wrapper,optimizer,data,inputs,loss,lp,kl,nll;gc.collect();torch.cuda.empty_cache()")
s=s.replace('workers=12','workers=40')
s=s.replace('fixed_perception_and_definitions=True,','fixed_perception_and_definitions=True,gpu_symbolic_planner=True,validation_workers_per_gpu=40,')
write(f/'train_history.py',s)
s=(f/'evaluate.py').read_text().replace('from planner import Program,plan,audit','from gpu_planner import Program,plan,audit,extend_batch')
s=s.replace('requests=[]\n','requests=[];prepared=[]\n',1)
marker='                        if history_predictor is None:'
assert marker in s
s=s.replace(marker,"""                        prepared.append((i,observed,observed_before_renewal,renew,probability))
                    extend_batch([programs[i] for i in active],32-step,device)
                    for i,observed,observed_before_renewal,renew,probability in prepared:
"""+marker,1)
write(f/'evaluate.py',s)
s=(f/'supervisor.py').read_text().replace('compact_history_v73',names[1]);write(f/'supervisor.py',s)

s=(BASE/'launch_parallel_v72_v73.py').read_text().replace('parallel_prefix_v72','parallel_prefix_v75').replace('288','768')
s=s.replace('continued_evidence_v72',names[0]).replace('compact_history_v73',names[1])
write(BASE/'launch_parallel_v75_v76.py',s)
s=(BASE/'deploy_parallel_v72_v73.py').read_text().replace('continued_evidence_v72',names[0]).replace('compact_history_v73',names[1]).replace('v72_v73','v75_v76')
write(BASE/'deploy_parallel_v75_v76.py',s)
print('Prepared larger768-example independent branches, GPU symbolic evaluation, measured batch sizing.')
