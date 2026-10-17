"""Training-only information-loss diagnostic; neither probe is a controller.

Actual later images are used to compare representations after recorded execution,
never to correct a symbolic rollout. Both probes predict the same learned IDs.
"""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld,Config
from predictive_nesy.subgoal_data import TransitionBank
from predictive_nesy.data import sha

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'diagnostics/transition_information_v12';assert not out.exists()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}[7]<300
out.mkdir();torch.set_num_threads(2);torch.manual_seed(3217);began=time.time()
path=root/'run_definition_world_v12/checkpoint_001600.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
assert sha(path)==json.loads(path.with_suffix('.COMMITTED.json').read_text())['sha256']
m=DefinitionConditionedWorld(Config(**saved['model_config'])).cuda().eval();m.load_state_dict(saved['model']);m.requires_grad_(False)
parts={k:[] for k in ['visual','prototype','context','target']};episode_ids=[]
with torch.no_grad():
    for bank_index,entry in enumerate(saved['manifest']['transition_banks']):
        bank=TransitionBank(entry['path'],saved['manifest']['data']['train_layouts'])
        for (start,offset),group in sorted(bank.by_key.items()):
            for pos in range(0,len(group),64):
                selected=group[pos:pos+64];rows=[]
                for name,t in selected:
                    d=bank.episodes[name];begin=max(0,start-10)
                    rows.append(dict(task=int(d['task']),history=d['codes'][begin:start+1],
                        previous_actions=np.array([int(d['actions'][j-1]) if j else 17 for j in range(begin,start+1)]),
                        prefix=d['codes'][start:t+1],successors=d['successors'][t]))
                d={k:torch.as_tensor(np.asarray([r[k] for r in rows]),device='cuda').long() for k in rows[0]}
                lengths=torch.full((len(rows),),d['history'].shape[1],device='cuda',dtype=torch.long)
                p=m.generate(d['task'],d['history'],d['previous_actions'],lengths,start=start)
                observed,memory,previous,codes=m.encode_sequence(p,d['prefix'])
                ids=observed[:,-1].argmax(-1)
                visual=F.normalize(previous+m.memory_projection(memory),dim=-1)
                prototype=F.normalize(p['definitions'][torch.arange(len(rows),device='cuda'),ids],dim=-1)
                future=m.observation(m.embed(d['successors'])).flatten(0,1)
                next_memory,next_feature,_=m.advance(memory.repeat_interleave(17,0),previous.repeat_interleave(17,0),
                    codes.repeat_interleave(17,0),future,d['successors'].flatten(0,1))
                feature=F.normalize(next_feature+m.memory_projection(next_memory),dim=-1)
                definitions=F.normalize(p['definitions'],dim=-1).repeat_interleave(17,0)
                target=torch.einsum('bd,bkd->bk',feature,definitions).argmax(-1).reshape(len(rows),17)
                for k,v in dict(visual=visual,prototype=prototype,context=m.context_norm(p['context_raw']),target=target).items():parts[k].append(v.cpu())
                episode_ids.extend(name for name,t in selected)
        print(json.dumps(dict(event='cached',bank=bank_index,states=len(episode_ids))),flush=True)
data={k:torch.cat(v).cuda() for k,v in parts.items()}
held=np.asarray([int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%5==0 for name in episode_ids])
fit=torch.tensor(np.flatnonzero(~held),device='cuda');hold=torch.tensor(np.flatnonzero(held),device='cuda')
assert not(set(np.asarray(episode_ids)[~held])&set(np.asarray(episode_ids)[held]))

class Predictor(nn.Module):
    def __init__(self):
        super().__init__();self.source=nn.Linear(256,512);self.context=nn.Linear(128,512);self.action=nn.Embedding(17,512)
        self.out=nn.Sequential(nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,128))
    def forward(self,x,c):return self.out(self.source(x)[:,None]+self.context(c)[:,None]+self.action.weight[None])

models={name:Predictor().cuda() for name in ['prototype','visual']}
models['visual'].load_state_dict(models['prototype'].state_dict())
opts={k:torch.optim.AdamW(v.parameters(),lr=3e-4,weight_decay=1e-4) for k,v in models.items()}
def evaluate(name,selection):
    correct=torch.zeros(17,device='cuda');loss=0.
    with torch.no_grad():
        for begin in range(0,len(selection),256):
            ix=selection[begin:begin+256];logits=models[name](data[name][ix],data['context'][ix]);target=data['target'][ix]
            correct+=(logits.argmax(-1)==target).sum(0)
            loss+=float(F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='sum'))
    return dict(accuracy=float(correct.sum())/(len(selection)*17),movement_accuracy=float(correct[1:].sum())/(len(selection)*16),
        stop_accuracy=float(correct[0])/len(selection),cross_entropy=loss/(len(selection)*17))
history=[];train_start=time.time()
for step in range(1,1501):
    ix=fit[torch.randint(len(fit),(512,),device='cuda')]
    for name,model in models.items():
        logits=model(data[name][ix],data['context'][ix]);loss=F.cross_entropy(logits.flatten(0,1),data['target'][ix].flatten())
        opts[name].zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opts[name].step()
    if step%100==0:print(json.dumps(dict(event='training',update=step,seconds=time.time()-train_start)),flush=True)
    if step%300==0:
        scores={name:dict(fit=evaluate(name,fit),held=evaluate(name,hold)) for name in models}
        history.append(dict(update=step,models=scores));print(json.dumps(dict(event='evaluation',**history[-1])),flush=True)
result=dict(training_only=True,post_execution_diagnostic=True,no_model_deployed=True,source=str(path),source_sha256=sha(path),
    fit_states=len(fit),held_states=len(hold),split='By episode, same split as v12 world fit; source checkpoint previously saw all banks',
    targets='Identical automatically learned successor IDs for both probes',
    inputs='Both receive boundary context and action ID; prototype uses only the current categorical state definition, visual uses its pre-quantization representation as an offline diagnostic',
    history=history,seconds=time.time()-began,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(event='complete',**result)),flush=True)
