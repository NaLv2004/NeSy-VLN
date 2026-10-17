"""Compare actual pretrained features and learned bits on identical training photos."""
import os,sys,time,json,random,collections
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_vision_v8')
sys.path.insert(0,str(root));os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
import torch.nn.functional as F
from PIL import Image
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,image_tensor,atomic
from nesy_vla.model import NavigationModel
torch.set_num_threads(2);torch.manual_seed(917);torch.cuda.set_per_process_memory_fraction(.24)
cfg=read(root/'configs/frozen_qwen.json');setup(cfg);bank=DataBank(cfg)
ids=random.Random(917).sample(bank.demo_ids,24);selected=[]
for eid in ids:
    rows=bank.demos[eid]
    for index in sorted(set([0,len(rows)//2,len(rows)-1])):selected.append(rows[index])
images=[Image.open(bank.demo_image_path(r)).convert('RGB') for r in selected]
records=[];started=time.time()
for name in ['frozen_convnext','frozen_qwen']:
    cfg=read(root/'configs'/f'{name}.json');model=NavigationModel(cfg).to('cuda').eval()
    path=root/'runs'/name/'evaluations/step_000010/checkpoint';marker=read(path/'COMMITTED');load_checkpoint(path,model)
    inputs=torch.cat([image_tensor(im,cfg['image_size'],model.device) for im in images])
    raw=[];encoded=[]
    with torch.no_grad():
        for part in inputs.split(8):
            with torch.autocast('cuda',dtype=torch.bfloat16):raw.append(model.grounder.vision(part).float())
            encoded.append(model.grounder.encode(part))
        raw=torch.cat(raw).flatten(2).transpose(1,2);tokens=torch.cat(encoded)
        def describe(x):
            means=x.mean(1);normal=F.normalize(means,dim=-1);pairwise=normal@normal.T;mask=~torch.eye(len(x),device=x.device,dtype=torch.bool)
            return {'rms':float(x.square().mean().sqrt()),'between_image_std':float(means.std(0,unbiased=False).mean()),
                'within_image_patch_std':float(x.std(1,unbiased=False).mean()),'pooled_pair_cosine':float(pairwise[mask].mean())}
        info={'arm':name,'checkpoint':marker,'raw_features':describe(raw),'projected_tokens':describe(tokens),'goals':{}}
        for category in ['chair','table']:
            p=model.generator('Find the '+category+' and stop close to it.',[],[])
            query=model.grounder.query;queries=query.query_norm(p.definitions)[None].expand(len(images),-1,-1)
            evidence=query.evidence_norm(tokens)
            _,weights=query.attention(queries,evidence,evidence,need_weights=True)
            truths=[];actions=[];probs=[];hiddens=[]
            for i in range(len(images)):
                result=model.step(p,None,[],tokens=tokens[i:i+1]);truths.append(result['observation']);actions.append(int(result['probs'].argmax()));probs.append(result['probs'])
                hiddens.append(query(p.definitions[None],tokens[i:i+1]).squeeze(0))
            truth=torch.stack(truths);bits=truth>=.5;frequency=bits.float().mean(0)
            h=torch.stack(hiddens);prob=torch.stack(probs)
            info['goals'][category]={'variable_bits':int(((frequency>0)&(frequency<1)).sum()),'unique_codes':len(set(tuple(x) for x in bits.cpu().tolist())),
                'mean_truth_std':float(truth.std(0,unbiased=False).mean()),'truth_margin':float((truth-.5).abs().mean()),
                'action_counts':dict(collections.Counter(actions)),'action_probability_std':float(prob.std(0,unbiased=False).mean()),
                'attention_entropy_fraction':float(-(weights*weights.clamp_min(1e-9).log()).sum(-1).mean()/torch.tensor(weights.shape[-1]).log()),
                'attention_max_weight':float(weights.max(-1).values.mean()),'query_evidence_between_image_std':float(h.std(0,unbiased=False).mean()),
                'truths':truth.cpu().tolist(),'actions':actions,'bit_frequency':frequency.cpu().tolist()}
        records.append(info)
        print(json.dumps({'arm':name,'raw':info['raw_features'],'tokens':info['projected_tokens'],
            'goals':{k:{a:b for a,b in v.items() if a not in ['truths','actions','bit_frequency']} for k,v in info['goals'].items()}}),flush=True)
    del model,raw,tokens,inputs,p,truth,bits,prob,h,queries,evidence,weights,result
    torch.cuda.empty_cache()
    inputs=torch.cat([image_tensor(im,cfg['image_size'],'cuda') for im in images])
out=root/'grounding_audit_step10.json'
atomic(out,{'time':time.time(),'seconds':time.time()-started,'training_only':True,'frames':[{'episode':r['episode_id'],'image':r['image'],'action':r['action']} for r in selected],
    'results':records,'peak_gib':torch.cuda.max_memory_allocated()/2**30})
print(json.dumps({'saved':str(out),'seconds':time.time()-started}),flush=True)
