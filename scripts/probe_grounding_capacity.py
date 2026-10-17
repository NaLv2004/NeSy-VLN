"""CPU-only, training-demonstration diagnostic. Never used to control the agent.

Compare reachable actions of fixed generated rules with the same rules driven by
the existing visual predicate grounder. This is a memorization/capacity test,
not a navigation evaluation and not an alternative policy architecture.
"""
import argparse,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from nesy_vla.program import Program
from nesy_vla.perception import PredicateGrounder
from nesy_vla.logic import SoftLogic
from nesy_vla.runtime import read,image_tensor,atomic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--steps',type=int,default=160)
    parser.add_argument('--evidence-only',action='store_true')
    parser.add_argument('--trace',action='append',help='Explicit completed training demonstration folder; repeat to pool trajectories')
    parser.add_argument('--output',help='Independent diagnostic output JSON')
    parser.add_argument('--free-only',action='store_true',help='Diagnose fixed rule expressivity without loading a visual model')
    parser.add_argument('--learn-confidence',action='store_true',help='Allow confidence as well as truth to vary in the free-state probe')
    parser.add_argument('--spatial-encoding',action='store_true')
    a=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(11);run=Path(a.run)
    cfg=read(run/'config.json');k=cfg['program']['k'];n_actions=cfg['program']['a']
    folders=[Path(f) for f in a.trace] if a.trace else [run/'demonstration_traces/update_000001']
    rows=[];ps=[];images=[];episodes=[]
    for folder in folders:
        trace=read(folder/'trajectory.json')
        assert all(row['expert'] for row in trace['steps'])
        assert trace['episode_id'].startswith('P25_train_')
        programs={}
        for f in folder.glob('program_*.json'):
            d=read(f)['program'];version=d['version']
            programs[version]=Program.from_record(d)
        rows.extend(trace['steps']);ps.extend(programs[r['version']] for r in trace['steps'])
        images.extend(Image.open(folder/r['image']).convert('RGB') for r in trace['steps'])
        episodes.append({'episode':trace['episode_id'],'steps':len(trace['steps']),'source':str(folder)})
    n=len(rows)
    labels=torch.tensor([r['action'] for r in rows]);logic=SoftLogic(k,n_actions,cfg['program']['r'])
    def predictions(b,c):return torch.stack([logic.infer(p,x,y)['probs'] for p,x,y in zip(ps,b,c)])
    def metrics(probs):
        return {'accuracy':float((probs.argmax(-1)==labels).float().mean()),
                'nll':float(-probs[torch.arange(n),labels].clamp_min(1e-9).log().mean()),
                'predictions':probs.argmax(-1).tolist()}
    result={'episodes':episodes,'steps':n,'labels':labels.tolist(),'not_navigation_evaluation':True}
    logits=torch.nn.Parameter(torch.zeros(n,k));confidence_logits=torch.nn.Parameter(torch.full((n,k),.84729786))
    free_confidence=lambda:confidence_logits.sigmoid() if a.learn_confidence else torch.full((n,k),.7)
    opt=torch.optim.Adam([logits,confidence_logits] if a.learn_confidence else [logits],lr=.12)
    for _ in range(a.steps):
        probs=predictions(logits.sigmoid(),free_confidence())
        loss=-probs[torch.arange(n),labels].clamp_min(1e-9).log().mean()
        opt.zero_grad();loss.backward();opt.step()
    result['free_predicate_capacity']=metrics(predictions(logits.sigmoid(),free_confidence()))
    result['free_confidence_learned']=a.learn_confidence
    if a.free_only:
        if not a.output:raise ValueError('--free-only requires explicit --output')
        result['state']='complete';atomic(a.output,result)
        print(json.dumps({'output':a.output,'steps':n,'free_predicate_capacity':result['free_predicate_capacity']},indent=2))
        return
    grounder=PredicateGrounder(cfg['program']['d'],n_actions,cfg['imagenet_weights'],a.evidence_only,a.spatial_encoding).eval()
    marker=read(run/'latest/COMMITTED')
    checkpoint=torch.load(run/'latest'/marker['checkpoint_file'],map_location='cpu',weights_only=False)
    state={name[len('grounder.'):]:v for name,v in checkpoint['model'].items() if name.startswith('grounder.')}
    if a.evidence_only:
        state={k:v for k,v in state.items() if not k.startswith('query.')}
        missing=grounder.load_state_dict(state,strict=False)
        assert all(k.startswith('query.') for k in missing.missing_keys)
        assert not missing.unexpected_keys
    else:grounder.load_state_dict(state,strict=True)
    del checkpoint,state
    with torch.no_grad():
        tokens=grounder.encode(torch.cat([image_tensor(im,cfg['image_size'],'cpu') for im in images])).detach()
    definitions=torch.stack([p.definitions for p in ps])
    parameters=[*grounder.query.parameters(),*grounder.truth.parameters(),*grounder.reliability.parameters()]
    optimizer=torch.optim.Adam(parameters,lr=.0002)
    def forward():
        hidden=grounder.query(definitions,tokens)
        b=grounder.truth(hidden).squeeze(-1).sigmoid();c=grounder.reliability(hidden).squeeze(-1).sigmoid()
        return predictions(b,c),b,c
    result['grounding_initial']=metrics(forward()[0]);learning=[];start=time.time()
    output=Path(a.output) if a.output else run/('grounding_capacity_evidence_only.json' if a.evidence_only else 'grounding_capacity_probe.json')
    result['grounder']='evidence_only' if a.evidence_only else 'residual_decoder'
    result['spatial_encoding']=a.spatial_encoding
    atomic(output,dict(result,state='running'))
    for step in range(a.steps):
        try:probs,b,c=forward()
        except Exception as error:
            atomic(output,dict(result,state='failed',at_step=step,error=str(error),learning=learning))
            raise
        loss=-probs[torch.arange(n),labels].clamp_min(1e-9).log().mean()
        loss=loss+.5*(c.mean()-.7).square()+2.*(.12-b.std(0,unbiased=False)).clamp_min(0).mean()
        optimizer.zero_grad();loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(parameters,2.)
        if not torch.isfinite(grad):
            atomic(output,dict(result,state='failed',at_step=step,error='nonfinite_gradient',learning=learning))
            raise FloatingPointError('Nonfinite visual-grounder gradient')
        optimizer.step()
        if (step+1)%20==0:learning.append(dict(step=step+1,**metrics(forward()[0])))
    with torch.no_grad():
        probabilities,truth,confidence=forward()
        permuted=grounder.query(definitions,tokens.flip(0))
        shuffled=predictions(grounder.truth(permuted).squeeze(-1).sigmoid(),
            grounder.reliability(permuted).squeeze(-1).sigmoid())
        result['visual_dependency']={'reversed_images':metrics(shuffled),
            'image_swap_changed_actions':int((probabilities.argmax(-1)!=shuffled.argmax(-1)).sum()),
            'mean_truth_std':float(truth.std(0,unbiased=False).mean()),
            'mean_confidence_std':float(confidence.std(0,unbiased=False).mean())}
    result.update(state='complete',grounding_final=metrics(probabilities),learning=learning,seconds=time.time()-start)
    atomic(output,result)
    print(json.dumps({'output':str(output),'steps':n,'grounder':result['grounder'],
        'accuracy':result['grounding_final']['accuracy'],'visual_dependency':result['visual_dependency'],
        'seconds':result['seconds']},indent=2))


if __name__=='__main__':main()
