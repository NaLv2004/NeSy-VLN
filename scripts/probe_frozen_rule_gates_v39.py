"""Can the neural rule interpreter learn from fixed automatically learned tokens?"""
import collections,copy,fcntl,json,os,random,sys,time
from pathlib import Path
import torch
from PIL import Image
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,DataBank,image_tensor,atomic,sha
from nesy_vla.model import NavigationModel
from nesy_vla.symbols import tokenize,DiscreteState
from nesy_vla.outcome_supervision import SymbolicExecutionDecoder

torch.set_num_threads(2);torch.manual_seed(3900);device='cuda:0'
out=base/'frozen_rule_learning_v39';out.mkdir(exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists(),'Do not repeat an existing capacity probe'
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
delta=base/'execution_learning_v34/execution_feedback_diagnostic_weights.pt'
cfg=read(source.parent/'config.json');cfg['recovery_demonstration_root']=str(base/'data/recovery_demonstrations_v27')
setup(cfg);bank=DataBank(cfg);dataset=read(base/'execution_learning_v34/dataset.json')
model=NavigationModel(cfg).eval();load_checkpoint(source,model)
model.execution_decoder=SymbolicExecutionDecoder(cfg['program']['d'],cfg['program']['a'])
mismatch=model.load_state_dict(torch.load(delta,map_location='cpu',weights_only=True),strict=False)
assert not mismatch.unexpected_keys and all(k.startswith(('grounder.vision.','teacher.')) for k in mismatch.missing_keys)
del model.execution_decoder;model.requires_grad_(False);model.to(device)
lookup={(eid,r['image']):r for eid in bank.recovery_demo_ids for r in bank.demos[eid]}
result={'state':'encoding_fixed_symbols','time':time.time(),'pid':os.getpid(),'source_commit':read(source/'COMMITTED'),
    'delta_sha256':sha(delta),'fixed_and_development_used':False,'not_navigation_success':True,
    'only_trainable_parameters':'logic.action_rule_gate; encoder, definitions, premises, conclusions and embeddings are frozen',
    'programs_fixed_from_training_tasks':True,'arms':{}}
atomic(out/'result.json',result);rows=[];programs={}
with torch.no_grad():
    for category in ['chair','table']:
        p=model.generator('Find the '+category+' and stop close to it.',[],[],None).detach();p.neural_conjunction_gates=True;programs[category]=p
        selected=[r for r in dataset['examples'] if r['category']==category]
        for start in range(0,len(selected),32):
            batch=selected[start:start+32];images=[Image.open(bank.demo_image_path(lookup[(r['id'],r['image'])])).convert('RGB') for r in batch]
            tokens=model.grounder.encode(torch.cat([image_tensor(im,cfg['image_size'],device) for im in images]))
            h=model.grounder.query(p.definitions[None].expand(len(batch),-1,-1),tokens)
            states=tokenize(model.grounder.truth(h).squeeze(-1).sigmoid())
            rows.extend(dict(r,codes=code.detach().clone()) for r,code in zip(batch,states.codes))
logic=model.logic;logic.enable_action_gates();logic.to(device)
original=copy.deepcopy(logic.state_dict());del model;torch.cuda.empty_cache()
train=[r for r in rows if not r['heldout']];held=[r for r in rows if r['heldout']]
assert len(train)==609 and len(held)==155
atomic(out/'symbols.json',{'examples':[dict(r,codes=r['codes'].cpu().tolist()) for r in rows],
    'programs':{k:p.record() for k,p in programs.items()}})

def forward(batch):
    probs=[];targets=[];labels=[]
    for category,p in programs.items():
        chosen=[r for r in batch if r['category']==category]
        if not chosen:continue
        codes=torch.stack([r['codes'] for r in chosen]);states=DiscreteState(codes,codes.float())
        probs.append(logic.probabilities_for_states(p,states));targets.extend(r['label'] for r in chosen);labels.extend(r['outcomes'] for r in chosen)
    return torch.cat(probs),torch.tensor(targets,device=device),torch.tensor(labels,device=device)

@torch.no_grad()
def measure(data):
    p,y,labels=forward(data);choice=p.argmax(-1);moves=choice>0;indices=torch.arange(len(choice),device=device)[moves]
    turns=(y!=0)&(y!=4)&(y!=11)
    return {'count':len(data),'expert_accuracy':float((choice==y).float().mean()),
        'turn_recall':float((choice[turns]==y[turns]).float().mean()),
        'nll':float(-p.gather(1,y[:,None]).clamp_min(1e-9).log().mean()),
        'collision_rate':float(labels[indices,choice[moves]-1,0].float().mean()),
        'counts':dict(collections.Counter(choice.tolist()))}

result['initial']={'train':measure(train),'holdout':measure(held)}
for lr in [1e-3,3e-3]:
    logic.load_state_dict(original);parameters=list(logic.action_rule_gate.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01);rng=random.Random(3900)
    arm={'learning':[]};result['arms'][str(lr)]=arm;start=time.time()
    for step in range(1,1201):
        p,y,_=forward(rng.sample(train,128));loss=-p.gather(1,y[:,None]).clamp_min(1e-9).log().mean()
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5.)
        assert torch.isfinite(norm);optimizer.step()
        if step%300==0:
            row={'step':step,'seconds':time.time()-start,'train':measure(train),'holdout':measure(held)}
            arm['learning'].append(row);result.update(state='training',time=time.time());atomic(out/'result.json',result)
            print(json.dumps({'lr':lr,**row}),flush=True)
    for p in programs.values():p.neural_conjunction_gates=False
    arm['disabled_gate_holdout']=measure(held)
    for p in programs.values():p.neural_conjunction_gates=True
    torch.save(logic.state_dict(),out/f'logic_lr{lr}.pt')
result.update(state='complete',time=time.time());atomic(out/'result.json',result)
