"""Compare inverse-action evidence before/after quantization on real training frames."""
import json,os,sys,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_robust_v7'
sys.path.insert(0,str(root));os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,atomic,image_tensor
from nesy_vla.model import NavigationModel
torch.set_num_threads(2);torch.manual_seed(14)
cfg=read(root/'configs/robust_multieffect.json');setup(cfg);model=NavigationModel(cfg).eval();bank=DataBank(cfg)
all_ids=sorted(e for e in bank.demo_ids if len(bank.demos[e])>=5)
chosen=[all_ids[len(all_ids)*i//6] for i in range(6)]
results=[]
for step in [10,30]:
    checkpoint=root/f'runs/robust_multieffect/evaluations/step_{step:06d}/checkpoint';load_checkpoint(checkpoint,model)
    rows=[]
    with torch.no_grad():
        for eid in chosen:
            demo=bank.demos[eid][:5]
            # Use the same demonstration loader so image paths, feedback, and goal match training.
            from nesy_vla.policy import demonstrations_many
            original=bank.demos[eid];bank.demos[eid]=demo
            trajectory=demonstrations_many(model,bank,[eid])[0];bank.demos[eid]=original
            chunk=trajectory.chunks[0]
            program=model.generator(chunk['task'],chunk['images'],chunk['feedback'],chunk['previous'])
            for row in chunk['rows']:
                action=row['action']
                if action==0 or row['collision'] or row['feedback'][1]<=.01:continue
                tokens=model.grounder.encode(image_tensor(row['image'],cfg['image_size'],model.device))
                visual,confidence=model.grounder.observe(tokens,program.definitions,[])
                target,_=model.transition_target(program,row['next_image'],[])
                predictions,_,_=model.logic.transition_all(program,visual,confidence,model.grounder.retain_all(program.definitions))
                codes=(predictions>=.5).float();hard_target=(target>=.5).float()
                soft_error=(predictions-target[None]).square().mean(-1)
                hard_error=(codes-hard_target[None]).square().mean(-1)
                best=hard_error.min();ties=(hard_error==best)
                rows.append({'episode':eid,'action':action,'unique_next_symbol_states':len(torch.unique(codes,dim=0)),
                    'soft_inverse_action':int(soft_error.argmin()),'hard_inverse_action':int(hard_error.argmin()),
                    'soft_correct':int(soft_error.argmin())==action,'hard_correct':int(hard_error.argmin())==action,
                    'hard_best_ties':int(ties.sum()),'hard_tie_adjusted_credit':float(ties[action])/int(ties.sum()),
                    'soft_error_range':float(soft_error.max()-soft_error.min()),
                    'changed_target_bits':int(((visual>=.5)!=hard_target.bool()).sum()),
                    'predicted_changed_bits':int((codes[action]!=(visual>=.5).float()).sum())})
    summary={'checkpoint':str(checkpoint),'frames':len(rows),'all_actions_same_codes':sum(r['unique_next_symbol_states']==1 for r in rows),
        'soft_correct':sum(r['soft_correct'] for r in rows),'hard_correct':sum(r['hard_correct'] for r in rows),
        'mean_unique_next_states':sum(r['unique_next_symbol_states'] for r in rows)/len(rows),
        'tie_adjusted_hard_accuracy':sum(r['hard_tie_adjusted_credit'] for r in rows)/len(rows)}
    results.append({'summary':summary,'rows':rows});print(json.dumps(summary),flush=True)
result={'time':time.time(),'results':results,'qualification':'Training-only frozen checkpoint diagnosis. No validation samples used; no training modified.'}
atomic(base/'diagnostics_20261011/inverse_symbol_codes.json',result)
