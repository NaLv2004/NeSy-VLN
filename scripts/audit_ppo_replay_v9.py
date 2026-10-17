import json,os,sys,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_joint_v9');sys.path.insert(0,str(root));os.environ['CUDA_VISIBLE_DEVICES']='6'
import torch
from PIL import Image
from nesy_vla.runtime import read,setup,load_checkpoint,image_tensor,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.policy import EpisodePolicy
from nesy_vla.program import Program
from nesy_vla.training import train_chunk
torch.set_num_threads(2);torch.manual_seed(28);torch.cuda.set_per_process_memory_fraction(.25)
cfg=read(root/'configs/joint.json');setup(cfg);model=NavigationModel(cfg).to('cuda').eval();load_checkpoint(cfg['initialize_from'],model)
folder=root/'integration/traces/update_000001/rank0_0';trace=read(folder/'trajectory.json');source=trace['steps']
images=[Image.open(folder/r['image']).convert('RGB') for r in source]
programs={}
for p in folder.glob('program_*.json'):
    program=Program.from_record(read(p)['program']).map(lambda x:x.to('cuda'));programs[program.version]=program
policy=EpisodePolicy(model,trace['category']);original_errors=[]
with torch.no_grad():
    for i,row in enumerate(source):
        request=policy.prepare(images[i])
        if request is not None:policy.install(programs[row['version']])
        _,result,replay=policy.choose(images[i],action=row['action'],stochastic=False,prepared=True)
        original_errors.append({'step':i,'bits':int((result['symbol_codes'].cpu()!=torch.tensor(row['symbol_codes'])).sum()),
            'logp_error':abs(replay['old_logp']-row['old_logp'])})
        after=images[min(i+1,len(images)-1)]
        response={'feedback':{'collision':row['collision']},'done':row['done'],'reward':row['reward'],'metrics':{'success':row['success']}}
        policy.observe(replay,result,after,response)
        replay.update(old_logp=row['old_logp'],advantage=row['advantage'],return_=row['return'])
        replay['return']=row['return']
results=[];original_step=model.step
for mode in ['eval_single_regenerated','train_single_regenerated','train_batch_regenerated','train_batch_saved_program']:
    model.train(mode.startswith('train'));details=[]
    for chunk in policy.chunks:
        iterator=iter(chunk['rows'])
        def captured(*args,**kwargs):
            result=original_step(*args,**kwargs);row=next(iterator)
            probability=result['probs'][row['action']]
            details.append({'bits':int((result['symbol_codes'].cpu()!=torch.tensor(row['symbol_codes'])).sum()),
                'ratio':float((probability.log()-row['old_logp']).exp().detach()),
                'max_probability_error':float((result['probs'].detach().cpu()-torch.tensor(row['support'])/sum(row['support'])).abs().max())})
            return result
        model.step=captured
        tokens=model.grounder.encode(torch.cat([image_tensor(r['image'],cfg['image_size'],model.device) for r in chunk['rows']])) if 'batch' in mode else None
        loss,stats=train_chunk(model,chunk,False,program=chunk['program'] if mode.endswith('saved_program') else None,tokens=tokens)
        del loss,tokens
        model.step=original_step
    summary={'mode':mode,'mean_ratio':sum(d['ratio'] for d in details)/len(details),
        'max_ratio_error':max(abs(d['ratio']-1) for d in details),'changed_bits':sum(d['bits'] for d in details),
        'max_probability_error':max(d['max_probability_error'] for d in details)}
    results.append({'summary':summary,'rows':details});print(json.dumps(summary),flush=True)
record={'time':time.time(),'trajectory':str(folder),'original_replay':original_errors,'results':results}
atomic(root/'ppo_replay_audit.json',record)
