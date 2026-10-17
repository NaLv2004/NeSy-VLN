"""CPU gradient diagnostic on a saved training demonstration and pinned weights.

The saved generated programs are fixed, not regenerated. No optimizer step,
environment action, or running training state is modified.
"""
import argparse,copy,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from PIL import Image
from nesy_vla.model import NavigationModel
from nesy_vla.perception import PredicateGrounder
from nesy_vla.logic import SoftLogic
from nesy_vla.program import Program
from nesy_vla.runtime import read,atomic,image_tensor,sha
from nesy_vla.training import train_chunk


class FixedProgramModel(NavigationModel):
    def __init__(self,cfg,checkpoint):
        nn.Module.__init__(self);self.cfg=cfg
        p=cfg['program'];self.grounder=PredicateGrounder(p['d'],p['a'],None,
            cfg.get('evidence_only_grounder',False),cfg.get('spatial_encoding',False))
        self.teacher=copy.deepcopy(self.grounder).requires_grad_(False).eval()
        self.logic=SoftLogic(p['k'],p['a'],p['r'])
        for prefix,module in [('grounder.',self.grounder),('teacher.',self.teacher)]:
            module.load_state_dict({k[len(prefix):]:v for k,v in checkpoint['model'].items() if k.startswith(prefix)},strict=True)
        self.eval()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--trace',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(19)
    checkpoint_folder=Path(args.checkpoint);marker=read(checkpoint_folder/'COMMITTED')
    path=checkpoint_folder/marker['checkpoint_file'];assert sha(path)==marker['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False);cfg=state['config']
    model=FixedProgramModel(cfg,state);del state
    folder=Path(args.trace);trace=read(folder/'trajectory.json');rows=trace['steps']
    assert trace['episode_id'].startswith('P25_train_') and all(r['expert'] for r in rows)
    assert all(v not in trace['scene'] for v in ['layout_200','layout_201'])
    images=[Image.open(folder/r['image']).convert('RGB') for r in rows]
    tensors=torch.cat([image_tensor(im,cfg['image_size'],'cpu') for im in images])
    with torch.no_grad():
        tokens=model.grounder.encode(tensors).detach();teacher_tokens=model.teacher.encode(tensors).detach()
    programs={};meta={}
    for f in folder.glob('program_*.json'):
        record=read(f);data=record['program'];version=data['version'];meta[version]=record
        programs[version]=Program.from_record(data)
    total=len(rows);memory=[];start=0;terms={};all_obs=[];stats=[]
    while start<total:
        version=rows[start]['version'];end=start+1
        while end<total and rows[end]['version']==version:end+=1
        converted=[dict(row,image=images[i],next_image=images[min(i+1,total-1)]) for i,row in enumerate(rows[start:end],start)]
        chunk={'task':trace['task'],'images':images[:start+1],'feedback':meta[version]['feedback'],
               'previous':programs.get(version-1),'memory':list(memory),'rows':converted}
        following=torch.stack([teacher_tokens[min(i+1,total-1)] for i in range(start,end)])
        _,metrics=train_chunk(model,chunk,True,program=programs[version],tokens=tokens[start:end],
            next_tokens=following,return_loss_terms=True)
        for name,term in metrics.pop('_loss_terms').items():terms[name]=terms.get(name,0.)+term*(end-start)/total
        all_obs.append(metrics.pop('_observations'));stats.append(metrics)
        with torch.no_grad():
            for i in range(start,end):
                memory.append(model.grounder.remember(tokens[i:i+1],rows[i]['action'],torch.tensor(rows[i]['feedback'])))
            memory=memory[-16:]
        start=end
    values=torch.cat(all_obs);centered=values-values.mean(0);sd=(centered.square().mean(0)+1e-5).sqrt()
    terms['variance']=(.12-sd).clamp_min(0).mean()
    normalized=centered/sd.clamp_min(.03);covariance=normalized.T@normalized/len(values)
    terms['decorrelation']=(covariance-torch.diag(covariance.diag())).square().mean()
    names=[];parameters=[]
    for name,param in model.grounder.named_parameters():
        if name.startswith(('query.','truth.','reliability.','retention.','event.')):
            names.append(name);parameters.append(param)
    gradients={};norms={}
    for name,term in terms.items():
        if not term.requires_grad:continue
        grad=torch.autograd.grad(term*cfg['loss'].get(name,1.),parameters,retain_graph=True,allow_unused=True)
        flat=torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(parameters,grad)]).detach()
        gradients[name]=flat;norms[name]=float(flat.norm())
    bc=gradients['bc'];combined=sum(gradients.values());cosine=lambda a,b:float((a@b)/(a.norm()*b.norm()).clamp_min(1e-12))
    result={'episode':trace['episode_id'],'checkpoint':str(path),'checkpoint_sha256':marker['checkpoint_sha256'],
        'fixed_programs':True,'not_navigation_evaluation':True,'steps':total,'parameters':names,
        'terms':{name:float(term.detach()) for name,term in terms.items()},'weighted_gradient_norms':norms,
        'cosine_with_bc':{name:cosine(bc,g) for name,g in gradients.items()},
        'combined_cosine_with_bc':cosine(bc,combined),'predicate_std':float(sd.mean()),
        'chunk_statistics':stats}
    atomic(args.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ['parameters','chunk_statistics']},indent=2))


if __name__=='__main__':main()
