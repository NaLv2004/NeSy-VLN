"""Exact Boolean counterfactual search of frozen rules; never policy input."""
import argparse,collections,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.symbols import DiscreteState
parser=argparse.ArgumentParser();parser.add_argument('--operator',choices=['product','minimum'],default='product')
parser.add_argument('--output',default='conjunctive_reachability_v30.json');args=parser.parse_args()
torch.set_num_threads(2);torch.manual_seed(30)
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');cfg['program']['conjunction_operator']=args.operator
setup(cfg);model=NavigationModel(cfg).eval();load_checkpoint(source,model)
started=time.time();result={};k=cfg['program']['k'];actions=cfg['program']['a']
@torch.no_grad()
def probabilities(program,bits):
    return torch.cat([model.logic.probabilities_for_states(program,DiscreteState(b.long(),b.float())) for b in bits.split(256)])
def margins(probs,targets):
    log=probs.clamp_min(1e-30).log();other=log.clone();other.scatter_(1,targets[:,None],-torch.inf)
    return log.gather(1,targets[:,None]).squeeze(1)-other.max(1).values
with torch.no_grad():
    for category in ['chair','table']:
        p=model.generator('Find the '+category+' and stop close to it.',[],[],None).detach()
        random_bits=torch.randint(2,(8192,k));random_probs=probabilities(p,random_bits)
        counts=collections.Counter(random_probs.argmax(1).tolist())
        targets=torch.arange(actions).repeat_interleave(16)
        # Each target starts from its16best distinct random seeds.
        seeds=torch.cat([random_bits[margins(random_probs,torch.full((len(random_bits),),a)).topk(16).indices] for a in range(actions)])
        for iteration in range(32):
            before=probabilities(p,seeds);old=margins(before,targets)
            candidates=seeds[:,None].repeat(1,k,1)
            indices=torch.arange(k);candidates[:,indices,indices]=1-candidates[:,indices,indices]
            candidate_probs=probabilities(p,candidates.reshape(-1,k))
            score=margins(candidate_probs,targets.repeat_interleave(k)).reshape(-1,k)
            values,best=score.max(1);improve=values>old+1e-7
            if not improve.any():break
            seeds[improve]=candidates[torch.arange(len(seeds))[improve],best[improve]]
        final=probabilities(p,seeds);score=margins(final,targets).reshape(actions,16)
        best=score.argmax(1)+torch.arange(actions)*16
        heads=p.action_heads[model.logic.action_slice]
        result[category]={'random_state_greedy_counts':dict(counts),'random_states':8192,
            'rule_conclusion_greedy_counts':dict(collections.Counter(heads.argmax(1).tolist())),
            'reachable_actions_found':int((final[best].argmax(1)==torch.arange(actions)).sum()),
            'exact_coordinate_search_iterations':iteration+1,
            'actions':[{'target':a,'greedy':int(final[i].argmax()),'probability':float(final[i,a]),
                'log_probability_margin':float(score[a].max()),'bits':seeds[i].tolist(),
                'largest_rule_conclusion_probability':float(heads[:,a].max())} for a,i in enumerate(best.tolist())]}
        actual=[r for r in read(base/'grounding_causality_v29.json')['examples'] if r['category']==category]
        actual_probs=probabilities(p,torch.tensor([r['codes'] for r in actual]))
        result[category]['static_training_frame_check']={
            'frames':len(actual),'greedy_counts':dict(collections.Counter(actual_probs.argmax(1).tolist())),
            'expert_action_agreement':float((actual_probs.argmax(1)==torch.tensor([r['action'] for r in actual])).float().mean())}
record={'time':time.time(),'source_commit':read(source/'COMMITTED'),'results':result,'seconds':time.time()-started,
    'conjunction_operator':args.operator,
    'only_synthetic_boolean_states':True,'no_training_or_navigation':True,
    'qualification':'Found states establish executable rule capacity only; they need not correspond to reachable observations. Failed search is not a proof of impossibility.'}
atomic(base/args.output,record)
print(json.dumps({**{k:v for k,v in record.items() if k!='results'},'results':{
    cat:{**{k:v for k,v in row.items() if k!='actions'},'reachable_action_ids':[r['target'] for r in row['actions'] if r['target']==r['greedy']]}
    for cat,row in result.items()}},indent=2),flush=True)
