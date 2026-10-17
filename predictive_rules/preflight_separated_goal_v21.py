import inspect,json,os,random,subprocess,time
from pathlib import Path
import torch
from predictive_nesy.raw_boundary_world import RawBoundaryWorld,Config
from predictive_nesy.separated_goal_world import SeparatedGoalWorld
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
from predictive_nesy.data import sha
from predictive_nesy.separated_goal_objectives import objective
from train_separated_goal_v21 import TrainingModule

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'preflight_separated_goal_v21';assert not out.exists()
assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in usage.splitlines()}[7]<300
torch.set_num_threads(2);torch.manual_seed(3919);rng=random.Random(3919)
source=root/'run_replayed_boundary_v20/checkpoint_000600.pt';assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
saved=torch.load(source,map_location='cpu',weights_only=False);old=saved['manifest']
base=RawBoundaryWorld(Config()).cuda().eval();base.load_state_dict(saved['model']);base.requires_grad_(False)
model=SeparatedGoalWorld(Config()).cuda();state=model.state_dict();state.update(saved['model'])
for n in list(state):
    if n.startswith('goal_'):state[n]=saved['model']['raw_'+n[len('goal_'):]].clone()
model.load_state_dict(state);prefixes=('definition_rules.','raw_','subgoal.','goal_')
for n,p in model.named_parameters():p.requires_grad_(n.startswith(prefixes))
frozen={n:v.cpu().clone() for n,v in model.state_dict().items() if not n.startswith(prefixes)}
features=ReplayedBoundaryFeatures(root/'prequant_visual_features_v18',root/'replayed_boundary_features_v20','cuda')
windows,_=load_windows(old['data']['source'],old['data']['train_layouts']);selected=rng.choices([w for w in windows if w['start']==10],k=512)
d=batch(selected,'cuda');raw,available=features.demonstrations(selected)
with torch.no_grad():
    args=(d['task'],d['history'],d['previous_actions'],d['history_length'])
    a=base.generate(*args,start=10,boundary_features=raw);b=model.generate(*args,start=10,boundary_features=raw)
    identity={k:bool(torch.equal(a[k],b[k])) for k in a if isinstance(a[k],torch.Tensor)}
assert all(identity.values());del a,b,base
# Measure routing using the very same objective tensors used by the trainer.
code=inspect.getsource(objective);assert code.count('return loss,metrics')==1
ns=dict(objective.__globals__);exec(compile(code.replace('return loss,metrics','return loss,metrics,losses'),'<routing-preflight>','exec'),ns)
small=batch(selected[:32],'cuda');sr,sa=features.demonstrations(selected[:32]);model.train()
with torch.autocast('cuda',dtype=torch.bfloat16):
    p=model.generate(small['task'],small['history'],small['previous_actions'],small['history_length'],start=10,boundary_features=sr)
    loss,_,parts=ns['objective'](model,small,proof_from_predicted_state=True,program=p)
world_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('definition_rules.','raw_'))]
goal_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('subgoal.','goal_'))]
def grad_sum(loss,parameters):
    gs=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return sum(float(g.abs().sum()) for g in gs if g is not None)
goal_loss=parts['subgoal_prediction']+parts['proof_action_imitation'];world_loss=2*parts['one_step_dynamics']
routing=dict(goal_loss_to_world=grad_sum(goal_loss,world_params),world_loss_to_goal=grad_sum(world_loss,goal_params),
    goal_loss_to_goal=grad_sum(goal_loss,goal_params),world_loss_to_world=grad_sum(world_loss,world_params))
assert routing['goal_loss_to_world']==0 and routing['world_loss_to_goal']==0
assert routing['goal_loss_to_goal']>0 and routing['world_loss_to_world']>0
del p,loss,parts,goal_loss,world_loss,small,sr,sa
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4);wrapper=TrainingModule(model)
d['boundary_features']=raw;d['boundary_available']=available;begin=time.time()
with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=wrapper(d,'demonstrations')
loss.backward();del d,loss
bank=TransitionBank(old['transition_banks'][1]['path'],old['data']['train_layouts']);d=bank.sample(rng,128,'cuda')
d['boundary_features'],d['boundary_available']=features.transitions(d,bank.directory);assert d['boundary_available'].all()
with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=wrapper(d,'transitions')
loss.backward();del d,loss
for params in [world_params,goal_params]:
    norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
opt.step();torch.cuda.synchronize();assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items())
out.mkdir();report=dict(passed=True,initial_program_bitwise_identical=identity,gradient_routing=routing,
    frozen_tensors_unchanged=len(frozen),real_update_seconds=time.time()-begin,memory_peak_gib=torch.cuda.max_memory_allocated()/2**30,
    demo_batch=512,transition_batch=128,source_sha256=sha(source),replay_manifest_sha256=sha(root/'replayed_boundary_features_v20/manifest.json'),
    no_runtime_contract_change=True,time=time.time(),metrics=metrics,transition_metrics=extra)
(out/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
