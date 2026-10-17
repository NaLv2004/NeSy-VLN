import json,os,random,subprocess,time
from pathlib import Path
import torch
from PIL import Image
from predictive_nesy.raw_boundary_world import RawBoundaryWorld,Config
from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
from predictive_nesy.data import sha
from predictive_nesy.environment import FrozenVision
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate
from train_raw_boundary_v19 import TrainingModule

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'preflight_raw_boundary_v19'
assert not out.exists();assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}[7]<300
torch.set_num_threads(2);torch.manual_seed(3919);rng=random.Random(3919)
source=root/'run_definition_world_v12/checkpoint_001600.pt'
assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
saved=torch.load(source,map_location='cpu',weights_only=False);old=saved['manifest']
base=DefinitionConditionedWorld(Config()).cuda().eval();base.load_state_dict(saved['model']);base.requires_grad_(False)
model=RawBoundaryWorld(Config()).cuda();state=model.state_dict();state.update(saved['model']);model.load_state_dict(state)
prefixes=('definition_rules.','subgoal.','raw_')
for name,p in model.named_parameters():p.requires_grad_(name.startswith(prefixes))
frozen={n:v.cpu().clone() for n,v in model.state_dict().items() if not n.startswith(prefixes)}
features=RawBoundaryFeatures(root/'prequant_visual_features_v18','cuda')
windows,_=load_windows(old['data']['source'],old['data']['train_layouts'])
chosen=rng.choices([w for w in windows if w['start']==10],k=512)
data=batch(chosen,'cuda');raw_features,available=features.demonstrations(chosen)
model.eval()
with torch.no_grad():
    a=base.generate(data['task'],data['history'],data['previous_actions'],data['history_length'],start=10)
    b=model.generate(data['task'],data['history'],data['previous_actions'],data['history_length'],start=10,boundary_features=raw_features)
    identity={key:bool(torch.equal(a[key],b[key])) for key in ['definitions','transitions','subgoal_logits','ground_memory','ground_last','ground_codes']}
assert all(identity.values());del a,b,base
model.train();wrapper=TrainingModule(model);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
data['boundary_features']=raw_features;data['boundary_available']=available
began=time.time()
with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=wrapper(data,'demonstrations')
loss.backward();total=float(loss.detach());del loss,data
bank=TransitionBank(old['transition_banks'][0]['path'],old['data']['train_layouts'])
data=bank.sample(rng,128,'cuda');data['boundary_features'],data['boundary_available']=features.transitions(data)
available_fraction=float(data['boundary_available'].float().mean())
with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=wrapper(data,'transitions')
loss.backward();total+=float(loss.detach());del loss,data
gradients={prefix:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) for prefix in ['definition_rules.','subgoal.','raw_delta.']}
assert all(v>0 for v in gradients.values())
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();torch.cuda.synchronize()
assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items())
duration=time.time()-began;peak=torch.cuda.max_memory_allocated()/2**30
vision=FrozenVision(old['frozen_vision']);gate=NeuralBoundaryGate([model,vision.vision]);encoder=RawBoundaryGraphEncoder(vision,gate)
rows=json.loads((Path(old['data']['source'])/'training_rows.json').read_text())
images=[Image.open(r['path']).convert('RGB') for r in rows[:4]]
try:
    with gate.boundary(0):canonical=vision.encode(images);joint,raw_joint=encoder.encode_joint(images)
    assert torch.equal(canonical,joint) and raw_joint.shape==(4,49,128)
    rejected=False
    try:encoder.encode_joint(images[:1])
    except RuntimeError:rejected=True
    assert rejected
finally:gate.close()
out.mkdir();result=dict(passed=True,initial_program_bitwise_identical=identity,canonical_visual_ids_exact=True,
    observation_outside_boundary_rejected=True,gradient_sums=gradients,frozen_tensors_unchanged=len(frozen),
    real_update_seconds=duration,loss=total,memory_peak_gib=peak,demo_batch=512,transition_batch=128,
    tested_transition_raw_availability=available_fraction,source_sha256=sha(source),time=time.time())
(out/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
