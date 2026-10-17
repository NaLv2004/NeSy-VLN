"""Reproduce the captured failing batches and locate their nonfinite derivatives."""
import inspect,json,random,time,torch
from pathlib import Path
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy import visual_graph_subgoal_training as training
from predictive_nesy.visual_memory_subgoal_training import transition_objective
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'gradient_replay_v8'
torch.set_num_threads(2)
source=inspect.getsource(training.objective).replace('return loss,metrics','return loss,metrics,losses')
namespace=dict(training.__dict__);exec(source,namespace);objective=namespace['objective']
results=[]
for rank in range(3):
    saved=torch.load(out/f'failure_rank{rank}.pt',map_location='cpu',weights_only=False)
    model=VisualMemorySubgoalModel(Config(**saved['model_config'])).cuda();model.load_state_dict(saved['model']);model.train()
    args=saved['manifest']['train_args'];data_info=saved['manifest']['data']
    windows,_=load_windows(data_info['source'],data_info['train_layouts']);starts=[w['start'] for w in windows]
    by_start={s:[w for w in windows if w['start']==s] for s in set(starts)}
    rng=random.Random();rng.setstate(saved['replay_rng']);start=rng.choice(starts)
    d=batch(rng.choices(by_start[start],k=args['batch_size']),'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics,terms=objective(model,d,proof_from_predicted_state=True)
    term_stats={}
    for name,value in terms.items():
        grad=torch.autograd.grad(value,model.history.weight_ih_l0,retain_graph=True,allow_unused=True)[0]
        term_stats[name]=dict(loss=float(value),max_abs=None if grad is None else float(grad.abs().max()),
            finite=True if grad is None else bool(torch.isfinite(grad).all()))
    loss.backward();demo_grads={n:dict(max_abs=float(p.grad.abs().max()),finite=bool(torch.isfinite(p.grad).all())) for n,p in model.named_parameters() if p.grad is not None}
    del loss,terms,d
    model.zero_grad(set_to_none=True)
    sources=[TransitionBank(p,data_info['train_layouts']) for p in args['transitions']]
    bank=rng.choice([s for s in sources if s.keys]);d=bank.sample(rng,args['transition_batch_size'],'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics_cf=transition_objective(model,d)
    loss.backward();cf_grads={n:dict(max_abs=float(p.grad.abs().max()),finite=bool(torch.isfinite(p.grad).all())) for n,p in model.named_parameters() if p.grad is not None}
    row=dict(rank=rank,start=start,terms=term_stats,demonstration_gradients=demo_grads,counterfactual_gradients=cf_grads)
    results.append(row);print(json.dumps(dict(rank=rank,start=start,terms=term_stats,bad_demo=[n for n,v in demo_grads.items() if not v['finite']],bad_cf=[n for n,v in cf_grads.items() if not v['finite']])),flush=True)
    del saved,model,loss,d;torch.cuda.empty_cache()
(out/'gradient_terms.json').write_text(json.dumps(results,indent=2))
