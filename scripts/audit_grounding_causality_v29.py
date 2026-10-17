"""Frozen train-only interventions: image -> discrete facts -> rule actions."""
import json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from PIL import Image
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,DataBank,image_tensor,atomic,sha
from nesy_vla.model import NavigationModel
from nesy_vla.symbols import DiscreteState
torch.set_num_threads(2);torch.manual_seed(29)
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');setup(cfg);model=NavigationModel(cfg).eval();load_checkpoint(source,model)
bank=DataBank(cfg);examples=[]
for category in ['chair','table']:
    seen=set()
    for eid in bank.demo_ids:
        episode=bank.by_id[eid]
        if episode['category']!=category or episode['scene'] in seen:continue
        seen.add(episode['scene']);rows=bank.demos[eid]
        for index in sorted({0,len(rows)//2,len(rows)-1}):
            row=rows[index];examples.append({'episode':eid,'scene':episode['scene'],'category':category,
                'index':index,'action':row['action'],'row':row})
        if len(seen)==8:break
# This diagnostic uses initial generated programs for a controlled image
# intervention; it is neither closed-loop navigation nor an expert-policy score.
print(json.dumps({'state':'selected','examples':len(examples),'source_sha':read(source/'COMMITTED')['checkpoint_sha256']}),flush=True)
results=[]
with torch.no_grad():
    # DataBank resolves recorded image roots and immutable row paths.
    for category in ['chair','table']:
        selected=[e for e in examples if e['category']==category]
        task='Find the '+category+' and stop close to it.'
        program=model.generator(task,[],[],None)
        rgb=[Image.open(bank.demo_image_path(e['row'])).convert('RGB') for e in selected]
        tokens=model.grounder.encode(torch.cat([image_tensor(im,cfg['image_size'],'cpu') for im in rgb]))
        blank=Image.new('RGB',rgb[0].size,(128,128,128))
        blank_tokens=model.grounder.encode(image_tensor(blank,cfg['image_size'],'cpu'))
        actual=[model.step(program,im,[],tokens=tokens[i:i+1]) for i,im in enumerate(rgb)]
        empty=model.step(program,blank,[],tokens=blank_tokens)
        for index,(example,result) in enumerate(zip(selected,actual)):
            other=actual[(index+3)%len(actual)]
            bits=result['symbol_codes'];counter=bits.repeat(len(bits),1)
            diagonal=torch.arange(len(bits));counter[diagonal,diagonal]=1-counter[diagonal,diagonal]
            probs=model.logic.probabilities_for_states(program,DiscreteState(counter,counter.float()))
            action=int(result['probs'].argmax())
            record={k:v for k,v in example.items() if k not in ['row','image']}
            record.update(codes=bits.tolist(),truth=result['observation'].tolist(),greedy=action,
                image_swap_bit_fraction=float((bits!=other['symbol_codes']).float().mean()),
                image_swap_greedy_changed=action!=int(other['probs'].argmax()),
                image_swap_probability_tv=float((result['probs']-other['probs']).abs().sum()/2),
                blank_bit_fraction=float((bits!=empty['symbol_codes']).float().mean()),
                blank_greedy_changed=action!=int(empty['probs'].argmax()),
                one_bit_greedy_changed=int((probs.argmax(-1)!=action).sum()),
                one_bit_mean_tv=float((probs-result['probs'][None]).abs().sum(-1).mean()/2),
                all_bit_flip_tv=float((model.logic.probabilities_for_states(program,
                    DiscreteState((1-bits)[None],(1-bits).float()[None]))[0]-result['probs']).abs().sum()/2))
            results.append(record)
truth=torch.tensor([r['truth'] for r in results]);aliases={}
for row in results:aliases.setdefault((row['category'],tuple(row['codes'])),[]).append(row)
summary={key:sum(float(r[key]) for r in results)/len(results) for key in [
    'image_swap_bit_fraction','image_swap_greedy_changed','image_swap_probability_tv','blank_bit_fraction',
    'blank_greedy_changed','one_bit_greedy_changed','one_bit_mean_tv','all_bit_flip_tv']}
summary.update(examples=len(results),unique_task_states=len(aliases),
    conflicting_label_states=sum(len({r['action'] for r in group})>1 for group in aliases.values()),
    examples_in_conflicting_states=sum(len(group) for group in aliases.values() if len({r['action'] for r in group})>1),
    truth_near_threshold_fraction=float(((truth-.5).abs()<.05).float().mean()),
    mean_truth_std=float(truth.std(0,unbiased=False).mean()))
record={'time':time.time(),'source':str(source),'source_commit':read(source/'COMMITTED'),
    'train_only':True,'initial_programs_only':True,'no_optimizer_update':True,'navigation_success_measured':False,
    'summary':summary,'examples':results}
assert sha(source/read(source/'COMMITTED')['checkpoint_file'])==record['source_commit']['checkpoint_sha256']
atomic(base/'grounding_causality_v29.json',record);print(json.dumps(summary,indent=2),flush=True)
