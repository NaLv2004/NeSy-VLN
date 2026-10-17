"""Training-only capacity audit before fitting or deploying a goal pointer.

An actual future observation supplies an ORACLE goal label in this diagnostic.
No oracle goal is deployed. Failure here rejects this program vocabulary rather
than masking it with navigation tuning or replacing an executable goal.
"""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.local_visual_program import LocalVisualProgram,LocalConfig,label_distribution
from predictive_nesy.discrete_visual_world import WorldConfig
from predictive_nesy.subgoal_data import load_windows
from predictive_nesy.graph_subgoal_interpreter import Program,plan
from predictive_nesy.data import sha

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'diagnostics/local_visual_program_v17';assert not out.exists()
assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}[7]<300
torch.set_num_threads(2);torch.manual_seed(3717)
source=root/'run_discrete_rollout_world_v16/checkpoint_000400.pt'
assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
saved=torch.load(source,map_location='cpu',weights_only=False);dm=saved['manifest']['data']
windows,provenance=load_windows(dm['data']['source'],dm['data']['train_layouts'])
held=np.asarray([int(hashlib.sha256(w['episode_id'].encode()).hexdigest()[:8],16)%5==0 for w in windows])
proposal_rows=[w['actions'] for w,h in zip(windows,held) if not h and w['action_valid'].all()]
proposals=np.unique(np.stack(proposal_rows),axis=0)
model=LocalVisualProgram(WorldConfig(**saved['model_config']),torch.tensor(proposals)).cuda().eval()
model.world.load_state_dict(saved['model']);model.requires_grad_(False)
rng=np.random.default_rng(3717);selections=[]
for name,mask in [('fit',~held),('held',held)]:
    ids=np.flatnonzero(mask);selections.extend((name,int(i)) for i in rng.choice(ids,64,replace=False))
out.mkdir();np.save(out/'proposal_actions.npy',proposals);rows=[];began=time.time()
with torch.inference_mode():
    for split,index in selections:
        w=windows[index];current=torch.tensor(w['targets'][:1],device='cuda')
        started=time.perf_counter();definitions,valid,ground=model.definitions(current)
        rules=model.rules(definitions,valid)
        compiler_seconds=time.perf_counter()-started
        target=torch.tensor(w['targets'][w['goal_index']][None],device='cuda')
        distribution,coverage=label_distribution(definitions,valid,target)
        goal=int(distribution.argmax(-1));n=int(valid.sum());s=int(ground[0])
        p=Program(rules[0,:n,:,:n].cpu().numpy().copy(),definitions[0,:n].float().cpu().numpy().copy(),
                  np.array([goal],np.int64),int(w['start'])).validate()
        route=plan(p,np.array([s]),horizon=min(10,32-w['start']))
        a=w['actions'][w['action_valid']].tolist();state=s;follow=[];grounded=[]
        for t,action in enumerate(a):
            state=int(p.transitions[state,action].argmax());follow.append(state)
            lab,_=label_distribution(definitions,valid,torch.tensor(w['targets'][t+1][None],device='cuda'))
            grounded.append(int(lab.argmax(-1)))
        rows.append(dict(episode_id=w['episode_id'],start=w['start'],split=split,category=w['task'],
            distinct_states=n,oracle_endpoint_patch_match=float(coverage),initial_equals_oracle_goal=s==goal,
            found=route['found'],planned_actions=route['actions'],teacher_actions=a,
            first_action_match=bool(route['actions'] and route['actions'][0]==a[0]),
            teacher_endpoint_match=bool(follow[-1]==goal),teacher_autonomous_matches=sum(x==y for x,y in zip(follow,grounded)),
            teacher_steps=len(a),premature_stop=bool(0 not in a and route['actions'] and route['actions'][-1]==0),
            compiler_seconds=compiler_seconds,oracle_goal_ties=int((distribution>0).sum())))
        if len(rows)%16==0:print(json.dumps(dict(event='capacity_audit',completed=len(rows),seconds=time.time()-began)),flush=True)
summary={}
for split in ['fit','held']:
    group=[r for r in rows if r['split']==split]
    summary[split]=dict(windows=len(group),found=sum(r['found'] for r in group),first_action_match=sum(r['first_action_match'] for r in group),
        teacher_endpoint_match=sum(r['teacher_endpoint_match'] for r in group),
        autonomous_matches=sum(r['teacher_autonomous_matches'] for r in group),teacher_steps=sum(r['teacher_steps'] for r in group),
        mean_goal_visual_match=float(np.mean([r['oracle_endpoint_patch_match'] for r in group])),
        mean_distinct_states=float(np.mean([r['distinct_states'] for r in group])),
        initial_equals_goal=sum(r['initial_equals_oracle_goal'] for r in group),
        premature_stop=sum(r['premature_stop'] for r in group),nonterminal_windows=sum(0 not in r['teacher_actions'] for r in group),
        median_program_seconds=float(np.median([r['compiler_seconds'] for r in group])))
result=dict(training_only=True,no_navigation_validation=True,no_neural_actor=True,no_value_table=True,no_maps=True,
    oracle_future_goal_only_for_diagnostic=True,no_model_deployed=True,source=str(source),source_sha256=sha(source),
    proposals=len(proposals),proposal_source='Unique ten-action strings from fit episodes only; independent of current task and future frames',
    data=provenance,summary=summary,rows=rows,seconds=time.time()-began)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
