import inspect,json,random,torch
from pathlib import Path
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy import visual_graph_subgoal_training as training
from predictive_nesy.subgoal_data import load_windows,batch
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'gradient_replay_v8';torch.set_num_threads(2)
saved=torch.load(out/'failure_rank0.pt',map_location='cpu',weights_only=False)
source=inspect.getsource(training.objective).replace('return loss,metrics','return loss,metrics,losses')
namespace=dict(training.__dict__);exec(source,namespace);objective=namespace['objective']
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts']);starts=[w['start'] for w in windows]
rng=random.Random();rng.setstate(saved['replay_rng']);start=rng.choice(starts)
d=batch(rng.choices([w for w in windows if w['start']==start],k=640),'cuda')
def stat(t):
    if t is None:return dict(unused=True)
    return dict(finite=bool(torch.isfinite(t).all()),max_abs=float(t.abs().max()),dtype=str(t.dtype))
results=[]
for amp in [True,False]:
    model=VisualMemorySubgoalModel(Config(**saved['model_config'])).cuda();model.load_state_dict(saved['model']);records={};handles=[]
    def hook(name):
        def record(module,args,result):
            values=result if isinstance(result,tuple) else (result,)
            for i,t in enumerate(values):
                if isinstance(t,torch.Tensor) and t.requires_grad:
                    key=name+':'+str(i);records[key]=dict(forward=stat(t))
                    t.register_hook(lambda grad,k=key:records[k].update(backward=stat(grad)))
        return record
    for name,module in model.named_modules():
        if name in ['history','context_read','context_norm','definitions','transitions','observation','memory_projection','subgoal']:
            handles.append(module.register_forward_hook(hook(name)))
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):loss,metrics,terms=objective(model,d,proof_from_predicted_state=True)
    terms['teacher_path_goal'].backward()
    grads={n:stat(p.grad) for n,p in model.named_parameters() if p.grad is not None}
    result=dict(amp=amp,loss=float(terms['teacher_path_goal']),module_stats=records,parameter_stats=grads)
    results.append(result);print(json.dumps(dict(amp=amp,loss=result['loss'],module_stats=records,bad_params=[n for n,v in grads.items() if not v['finite']])),flush=True)
    for handle in handles:handle.remove()
    del terms,loss,model;torch.cuda.empty_cache()
(out/('backward_precision_'+Path.cwd().name+'.json')).write_text(json.dumps(results,indent=2))
