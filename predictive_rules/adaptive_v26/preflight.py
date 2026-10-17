"""One real GPU forward/backward and generation-contract preflight."""
import json,os,random,time
import torch
from torch.nn import functional as F
from adaptive_model import AdaptiveObservedWorld,Config
from hard_interpreter import export
from train import ROOT,Windows,prepare,observed_objective
from predictive_nesy.subgoal_data import batch,TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures
from predictive_nesy.selected_row_objectives import transition_objective

assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
torch.set_num_threads(2);torch.manual_seed(26161);rng=random.Random(26161)
began=time.time();source=ROOT/'run_state512_v25/checkpoint_003000.pt'
old=torch.load(source,map_location='cpu',weights_only=False)
model=AdaptiveObservedWorld(Config(**old['model_config'])).cuda()
missing,unexpected=model.load_state_dict(old['model'],strict=False)
assert not unexpected and all(k.startswith(('correction.','revision.')) for k in missing)
for name,p in model.named_parameters():p.requires_grad_(name.startswith(('definition_rules.','subgoal.','raw_','goal_','rule_evidence.','revision.','correction.')))
corpus=Windows(old['manifest']['data']['source'],old['manifest']['data']['train_layouts'])
features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cuda')
print(json.dumps(dict(stage='loaded',seconds=time.time()-began,windows=len(corpus.keys))),flush=True)
begin=time.time();start=7;keys=rng.choices(corpus.by_start[start],k=16);rows=[corpus.window(*key) for key in keys];data=batch(rows,'cuda')
p=prepare(model,data,features,rows)
with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics,labels=observed_objective(model,data,p)
loss.backward()
gradients={name:sum(float(t.grad.float().norm()) for n,t in model.named_parameters() if n.startswith(name) and t.grad is not None) for name in ['correction.','definition_rules.','subgoal.']}
assert torch.isfinite(loss) and all(v>0 for v in gradients.values()),gradients
model.zero_grad(set_to_none=True)
obs=labels[:,0];hidden=torch.zeros(16,128,device='cuda');goal=p['subgoal_logits'].argmax(-1).flatten()
logit,_=model.revision.step(obs,torch.randint(512,(16,),device='cuda'),data['previous_actions'][:,7],data['task'],torch.ones(16,device='cuda'),hidden,goal)
renewal=(logit.sigmoid()*torch.linspace(-2,2,16,device='cuda')).mean();renewal.backward()
gradients['revision.']=sum(float(t.grad.norm()) for n,t in model.named_parameters() if n.startswith('revision.') and t.grad is not None)
assert gradients['revision.']>0
model.zero_grad(set_to_none=True)
bank=TransitionBank(old['manifest']['transition_banks'][0]['path'],old['manifest']['data']['train_layouts'])
branches=bank.sample(rng,8,'cuda');raw,available=features.transitions(branches,bank.directory)
lengths=torch.full((8,),branches['history'].shape[1],device='cuda',dtype=torch.long)
with torch.autocast('cuda',dtype=torch.bfloat16):
    branchp=model.prepare(branches['task'],branches['history'],branches['previous_actions'],lengths,start=branches['start'],boundary_features=raw,boundary_available=available)
    bl,bm=transition_objective(model,branches,branchp)
bl.backward();assert torch.isfinite(bl)
model.zero_grad(set_to_none=True)
with torch.no_grad():
    one=[corpus.window(keys[0][0],3)];d=batch(one,'cuda');raw,available=features.demonstrations(one)
    generated=model.generate(d['task'],d['history'][:,:4],d['previous_actions'][:,:4],d['history_length'],start=3,boundary_features=raw,boundary_available=available)
    program=export(generated);assert program.start==3
    for predicted,observed in [(0,0),(511,511),(3,124),(252,47)]:
        selected=model.correction.selected(generated['definitions'],generated['context_raw'],torch.tensor([predicted],device='cuda'),torch.tensor([observed],device='cuda')).argmax(-1)
        assert int(selected)==int(program.correction[predicted,observed])
torch.cuda.synchronize()
print(json.dumps(dict(stage='passed',seconds=time.time()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30,gradients=gradients,metrics=metrics,branch_metrics=bm)),flush=True)
