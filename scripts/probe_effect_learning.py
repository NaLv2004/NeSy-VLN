"""Training-trace capacity experiment for learned symbolic effects, CPU only.

Predicate meanings, states and targets all come from the saved model. No target
geometry or hand-specified predicate semantics enter this diagnostic.
"""
import os,sys,json,argparse,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from nesy_vla.program import Program
from nesy_vla.neural_logic import NeuralSymbolicLogic


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--output',required=True)
    p.add_argument('--iterations',type=int,default=400);args=p.parse_args();torch.set_num_threads(2)
    folder=Path(args.folder);trace=json.loads((folder/'trajectory.json').read_text())
    rows=[r for r in trace['steps'] if r['action']!=0]
    assert rows and all(r['expert'] for r in rows) and trace['episode_id'].startswith('P25_train_')
    programs={}
    for f in folder.glob('program_*.json'):
        raw=json.loads(f.read_text())['program']
        q=Program.from_record(raw);programs[q.version]=q
    q=next(iter(programs.values()));k=len(q.definitions);a=q.action_heads.shape[-1]
    logic=NeuralSymbolicLogic(k,a,len(q.weights))
    logic.load_state_dict(torch.load(folder/'logic_snapshot.pt',map_location='cpu',weights_only=True));logic.requires_grad_(False)
    source=torch.tensor([r['belief'] for r in rows]);bits=source>=.5
    target=torch.tensor([r['observed_next_truth'] for r in rows]);changed=(bits!=(target>=.5))
    confidence=torch.ones(k);trials=[];started=time.time()
    for effect_kind,balanced in [('softmax',False),('softmax',True),('independent',True),('per_predicate',True)]:
        params={};all_parameters=[]
        for v,program in programs.items():
            effects=program.effects.clamp(1e-6,1-1e-6)
            if effect_kind=='per_predicate':
                add,delete=effects.chunk(2,-1)
                initial=effects.log()-torch.cat([1-add-delete,1-add-delete],-1).clamp_min(1e-6).log()
            else:initial=effects.log() if effect_kind=='softmax' else torch.logit(effects)
            params[v]=[torch.nn.Parameter(initial),torch.nn.Parameter(program.action_conditions.clamp_min(1e-6).log()),
                torch.nn.Parameter(torch.logit(program.weights.clamp(1e-5,1-1e-5))),torch.nn.Parameter(torch.zeros(a,k))]
            all_parameters.extend(params[v])
        optimizer=torch.optim.Adam(all_parameters,lr=.03);history=[]
        for iteration in range(args.iterations):
            current={}
            for v,program in programs.items():
                e,c,w,keep=params[v]
                if effect_kind=='per_predicate':
                    grouped=e.reshape(e.shape[0],2,k)
                    learned=torch.cat([torch.zeros_like(grouped[:,:1]),grouped],1).softmax(1)[:,1:].flatten(1)
                else:learned=e.softmax(-1) if effect_kind=='softmax' else e.sigmoid()
                current[v]=Program(program.definitions,program.premises,program.state_heads,program.action_heads,
                    learned,c.softmax(-1),w.sigmoid(),v,'per_predicate' if effect_kind=='per_predicate' else 'categorical')
            predicted=torch.stack([logic.transition(current[r['version']],b,confidence,r['action'],
                params[r['version']][3][r['action']].sigmoid())[0] for b,r in zip(source,rows)])
            losses=F.binary_cross_entropy(predicted.clamp(1e-6,1-1e-6),target,reduction='none')
            if balanced:
                pieces=[losses[mask].mean() for mask in [changed,~changed] if mask.any()]
                loss=torch.stack(pieces).mean()
            else:loss=losses.mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            if iteration==0 or (iteration+1)%100==0:
                right=(predicted.detach()>=.5)==(target>=.5)
                history.append({'iteration':iteration+1,'loss':float(loss.detach()),
                    'bit_accuracy':float(right.float().mean()),'changed_bit_accuracy':float(right[changed].float().mean()),
                    'unchanged_bit_accuracy':float(right[~changed].float().mean())})
        trials.append({'effect_kind':effect_kind,'balanced_change_loss':balanced,'history':history})
    result={'source':str(folder),'frames':len(rows),'changed_bits':int(changed.sum()),'total_bits':changed.numel(),
        'copy_accuracy':float((~changed).float().mean()),'trials':trials,'seconds':time.time()-started,
        'limitation':'Optimizes free program-effect tensors and retention on one training trace; capacity evidence, not navigation performance.'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
