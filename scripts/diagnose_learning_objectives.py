"""Train-only, fixed-checkpoint audit of gradients through actual discrete proofs.

This never updates a training process, supplies semantic predicate labels, or
evaluates navigation. RGB is cached only inside the perceptual module.
"""
import argparse, copy, json, os, random, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--synchronous-teacher',action='store_true')
    args=p.parse_args();sys.path.insert(0,args.root)
    from nesy_vla.runtime import read, setup, atomic, sha, DataBank, image_tensor, public_feedback, load_checkpoint
    from nesy_vla.model import NavigationModel
    from nesy_vla.training import train_chunk
    from nesy_vla.symbols import tokenize
    torch.set_num_threads(2);torch.manual_seed(1211);random.seed(1211)
    marker=read(Path(args.checkpoint)/'COMMITTED');path=Path(args.checkpoint)/marker['checkpoint_file']
    assert sha(path)==marker['checkpoint_sha256']
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);cfg=checkpoint['config'];setup(cfg)
    model=NavigationModel(cfg).eval();load_checkpoint(args.checkpoint,model);del checkpoint
    if args.synchronous_teacher:model.teacher.load_state_dict(model.grounder.state_dict(),strict=True)
    bank=DataBank(cfg);groups={}
    for eid in bank.demo_ids:
        ep=bank.by_id[eid]
        if eid in bank.paired_demo_ids:continue
        key=(ep['domain'],ep['category'],ep['distance_group'])
        if key[2] in ('near','far'):groups.setdefault(key,[]).append(eid)
    chosen=[sorted(ids)[len(ids)//2] for key,ids in sorted(groups.items())]
    assert len(chosen)>=4
    names=[];params=[]
    for name,param in model.named_parameters():
        if param.requires_grad and not name.startswith(('grounder.vision.','teacher.')):
            names.append(name);params.append(param)
    terms={};metrics=[];observations=[];stability=[];cached=[];started=time.time()
    for eid in chosen:
        source=bank.demos[eid][:10];task='Find the '+source[0]['category']+' and stop close to it.'
        all_rows=bank.demos[eid];images=[Image.open(bank.demo_image_path(row)).convert('RGB') for row in all_rows[:11]]
        with torch.no_grad():
            tensors=torch.cat([image_tensor(im,cfg['image_size'],'cpu') for im in images])
            tokens=model.grounder.encode(tensors).detach();next_tokens=model.teacher.encode(tensors).detach()
        converted=[]
        for i,row in enumerate(source):
            ni=min(i+1,len(images)-1);feedback,vector=public_feedback({'feedback':row.get('public_feedback',{})},images[i],images[ni])
            converted.append(dict(row,image=images[i],next_image=images[ni],feedback=vector,collision=feedback['collision']))
        chunk={'task':task,'images':[images[0]],'feedback':[],'previous':None,'memory':[],'rows':converted}
        program=model.generator(task,[],[])
        following=torch.stack([next_tokens[min(i+1,len(images)-1)] for i in range(len(source))])
        _,stats=train_chunk(model,chunk,True,program=program,tokens=tokens[:len(source)],next_tokens=following,return_loss_terms=True)
        for name,value in stats.pop('_loss_terms').items():terms[name]=terms.get(name,0.)+value/len(chosen)
        observations.append(stats.pop('_observations'));metrics.append(dict(episode=eid,steps=len(source),**stats))
        cached.append((chunk,tokens[:len(source)],following))
        with torch.no_grad():
            for i in range(len(source)):
                student,_=model.grounder.observe(tokens[i:i+1],program.definitions,[])
                teacher,_=model.teacher.observe(next_tokens[i:i+1],program.definitions,[])
                stability.append({'episode':eid,'step':i,'ema_current_bit_disagreement':float(((student>=.5)!=(teacher>=.5)).float().mean()),
                                  'near_threshold_fraction':float(((student-.5).abs()<.05).float().mean())})
    values=torch.cat(observations);centered=values-values.mean(0);sd=(centered.square().mean(0)+1e-5).sqrt()
    terms['variance']=(.12-sd).clamp_min(0).mean()
    normed=centered/sd.clamp_min(.03);cov=normed.T@normed/len(values)
    terms['decorrelation']=(cov-torch.diag(cov.diag())).square().mean()
    terms['symbol_usage']=(tokenize(values).values.mean(0)-.5).square().mean()
    groups={'generator':[], 'grounding':[], 'logic':[]}
    for i,name in enumerate(names):groups['generator' if name.startswith('generator.') else 'logic' if name.startswith('logic.') else 'grounding'].append(i)
    gradients={};weighted={};weights=cfg['loss']
    for name,term in terms.items():
        if not term.requires_grad or name.endswith('_metric'):continue
        grad=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
        gradients[name]={key:torch.cat([(torch.zeros_like(params[i]) if grad[i] is None else grad[i]).detach().flatten() for i in ids]) for key,ids in groups.items()}
        weighted[name]={key:vec*weights.get(name,1.) for key,vec in gradients[name].items()}
    cosine=lambda a,b:float(a.dot(b)/(a.norm()*b.norm()).clamp_min(1e-15))
    summary={}
    for key in groups:
        bc=weighted['bc'][key];combined=sum(g[key] for g in weighted.values())
        summary[key]={'bc_norm':float(bc.norm()),'combined_norm':float(combined.norm()),'combined_cosine_with_bc':cosine(bc,combined),
            'terms':{name:{'weighted_norm':float(g[key].norm()),'cosine_with_bc':cosine(bc,g[key])} for name,g in weighted.items()}}
    result={'time':time.time(),'seconds':time.time()-started,'checkpoint':str(path),'commit':marker,'episodes':chosen,
        'train_split_hash':bank.provenance['train_hash'],'fixed_split_hash':bank.provenance['fixed_hash'],
        'navigation_evaluation':False,'fixed_data_used':False,'same_frame_teacher_comparison':True,
        'synchronous_teacher':args.synchronous_teacher,
        'method':'First <=10 expert actions on 8 stratified training episodes; no photometric perturbation; cached encoder; actual generated rules and hard symbols.',
        'loss_weights':weights,'terms':{k:float(v.detach()) for k,v in terms.items()},'gradients':summary,
        'mean_teacher_bit_disagreement':sum(x['ema_current_bit_disagreement'] for x in stability)/len(stability),
        'mean_near_threshold_fraction':sum(x['near_threshold_fraction'] for x in stability)/len(stability),
        'per_frame_stability':stability,'metrics':metrics}
    atomic(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','per_frame_stability','loss_weights')},indent=2))


if __name__=='__main__':main()
