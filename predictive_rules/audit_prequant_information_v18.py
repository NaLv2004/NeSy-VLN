"""Compare coarse patch tokens with their continuous PRE-quantization inputs.

Both probes predict identical discrete successor IDs. Continuous visual vectors
are inputs at a hypothetical rule revision only, never state-transition targets.
This is offline diagnosis on training layouts, not an executable controller.
"""
import json,os,re,subprocess,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld,Config
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.data import sha

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'diagnostics/prequant_information_v18'
assert not out.exists();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}[3]<300
torch.set_num_threads(2);torch.manual_seed(3818);began=time.time()
cache=root/'prequant_visual_features_v18';cm=json.loads((cache/'manifest.json').read_text())
assert sha(cache/'features.npy')==cm['feature_sha256']
source=Path(cm['source_checkpoint']);assert sha(source)==cm['source_sha256']
saved=torch.load(source,map_location='cpu',weights_only=False);manifest=saved['manifest'];directory=Path(manifest['data']['source'])
rows=json.loads((directory/'training_rows.json').read_text());tokens=np.load(directory/'visual_tokens.npy');features=np.load(cache/'features.npy')
groups=defaultdict(list)
for index,row in enumerate(rows):groups[row['episode_id']].append(index)
windows,provenance=load_windows(directory,manifest['data']['train_layouts'])
model=DefinitionConditionedWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
parts={k:[] for k in ['prototype','context','action','target']};image_indices=[];layouts=[];episodes=[]
with torch.no_grad():
    for start in sorted(set(w['start'] for w in windows)):
        subset=[w for w in windows if w['start']==start]
        for begin in range(0,len(subset),128):
            selected=subset[begin:begin+128];d=batch(selected,'cuda');width=min(start+1,11)
            p=model.generate(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start)
            states=model.training_targets(p,d['targets'],d['actions']).squeeze(2).argmax(-1)
            for j,w in enumerate(selected):
                indices=groups[w['episode_id']];row0=rows[indices[0]]
                layout=int(re.search(r'P25_train_(\d+)_',row0.get('source_episode_id',row0['episode_id'])).group(1))
                count=int(w['action_valid'].sum());ids=states[j,:count]
                parts['prototype'].append(p['definitions'][j,ids].cpu());parts['context'].append(model.context_norm(p['context_raw'][j]).expand(count,-1).cpu())
                parts['action'].append(d['actions'][j,:count].cpu());parts['target'].append(states[j,1:count+1].cpu())
                image_indices.extend(indices[min(start+t,len(indices)-1)] for t in range(count))
                layouts.extend([layout]*count);episodes.extend([w['episode_id']]*count)
        print(json.dumps(dict(event='labels_cached',start=start,transitions=len(image_indices))),flush=True)
del model;torch.cuda.empty_cache()
data={k:torch.cat(v).cuda() for k,v in parts.items()}
data['visual']=torch.tensor(features[image_indices],device='cuda').float().flatten(1)
data['codes']=torch.tensor(tokens[image_indices],device='cuda').long()
hold=np.isin(layouts,[118,119,120,121]);fit=torch.tensor(np.flatnonzero(~hold),device='cuda');held=torch.tensor(np.flatnonzero(hold),device='cuda')
assert len(fit)>0 and len(held)>0 and not(set(np.asarray(episodes)[hold])&set(np.asarray(episodes)[~hold]))
dim=features.shape[-1]

class Predictor(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.embedding=nn.Embedding(49*64,dim);self.register_buffer('offset',torch.arange(49)*64)
        self.visual=nn.Linear(49*dim,512);self.common=nn.Linear(256+128,512);self.action=nn.Embedding(17,512)
        self.output=nn.Sequential(nn.LayerNorm(512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,128))
        if mode=='prequant':self.embedding.requires_grad_(False)
    def forward(self,ix):
        x=data['visual'][ix] if self.mode=='prequant' else self.embedding(data['codes'][ix]+self.offset).flatten(1)
        return self.output(self.visual(x)+self.common(torch.cat([data['prototype'][ix],data['context'][ix]],-1))+self.action(data['action'][ix]))

models={mode:Predictor(mode).cuda() for mode in ['tokens','prequant']}
models['prequant'].load_state_dict(models['tokens'].state_dict())
opts={k:torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01) for k,m in models.items()}

@torch.no_grad()
def evaluate(m,selection):
    m.eval();n=0;correct=0;movement=0;movement_correct=0;ce=0.
    for begin in range(0,len(selection),512):
        ix=selection[begin:begin+512];logits=m(ix);target=data['target'][ix];mask=data['action'][ix]!=0;match=logits.argmax(-1)==target
        n+=len(ix);correct+=int(match.sum());movement+=int(mask.sum());movement_correct+=int((match&mask).sum())
        ce+=float(F.cross_entropy(logits,target,reduction='sum'))
    return dict(transitions=n,accuracy=correct/n,movement_accuracy=movement_correct/max(movement,1),cross_entropy=ce/n)

history=[];out.mkdir();best={mode:-1 for mode in models};train_start=time.time()
for step in range(1,1201):
    ix=fit[torch.randint(len(fit),(1024,),device='cuda')]
    for mode,m in models.items():
        m.train();loss=F.cross_entropy(m(ix),data['target'][ix]);opts[mode].zero_grad();loss.backward()
        norm=nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opts[mode].step()
    if step%100==0:print(json.dumps(dict(event='training',update=step,seconds=time.time()-train_start)),flush=True)
    if step%200==0:
        scores={mode:dict(fit=evaluate(m,fit),held=evaluate(m,held)) for mode,m in models.items()}
        history.append(dict(update=step,models=scores));print(json.dumps(dict(event='evaluation',**history[-1])),flush=True)
        for mode,m in models.items():
            accuracy=scores[mode]['held']['movement_accuracy']
            if accuracy>best[mode]:
                best[mode]=accuracy;torch.save(dict(model=m.state_dict(),update=step,mode=mode,source=str(source),no_model_deployed=True),out/f'{mode}_best_probe.pt')
result=dict(training_only=True,no_navigation_validation=True,no_model_deployed=True,no_continuous_state_target=True,
    targets='Identical128-way successor IDs learned by the frozen v12 model',
    source=str(source),source_sha256=sha(source),feature_cache=cm,fit_transitions=len(fit),held_transitions=len(held),
    held_training_layouts=[118,119,120,121],source_representation_already_trained_on_these_layouts=True,
    input_comparison='Both receive identical source-state definition, boundary context and action ID; token input has learned patch-specific embeddings, prequant input has CNN/PCA vectors before64-way quantization',
    history=history,best_held_movement=best,seconds=time.time()-began,peak_gib=torch.cuda.max_memory_allocated()/2**30)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['history','feature_cache']}),flush=True)
