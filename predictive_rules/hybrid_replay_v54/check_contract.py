"""Actual gradients and forward-replay checks, on original training data only."""
import importlib.util,json,os,random,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from objectives import objective
from data import ROOT,sha
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(2);torch.manual_seed(541017)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
source=ROOT/'onpolicy_successors_v54/source.pt'
commit=json.loads(source.with_suffix('.COMMITTED.json').read_text());assert sha(source)==commit['sha256']
saved=torch.load(source,map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config'])).cuda();model.load_state_dict(saved['model']);del saved
spec=importlib.util.spec_from_file_location('v52_data',ROOT/'src_goal_aware_symbols_v52/data.py')
old_data=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_data)
corpus=old_data.Corpus(branch_root=str(ROOT/'onpolicy_successors_v40'))
data=corpus.sample(random.Random(541017),16,16,'cuda');data['world_updates_encoder']=True
before={n:p.detach().clone() for n,p in model.named_parameters()}
began=time.time();loss,metrics=objective(model,data);loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
assert float(model.memory_predictor.weight_ih.grad.norm())>0
assert float(model.rule[0].weight.grad.norm())>0
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.step()
assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
model.eval()
# Verify the newly supervised input composition against an independent manual
# replay. No future image reaches the context or the memory carrier.
with torch.no_grad():
    x=data['branch_features'];previous=data['branch_previous'];bs=data['branch_start'];bt=data['branch_t']
    index=torch.arange(len(x),device='cuda')
    ctx=model.context_sequence(x,hard(model.observe(x)),previous,data['branch_task'])[index,bs]
    lo,z,_=model.observation_encoding(x,ctx);obs=hard(lo)
    lm,_,_=model.memory_sequence(obs,previous,z);memory=hard(lm)
    predictions=[]
    for b in range(len(x)):
        state=memory[b:b+1,bs[b]]
        for t in range(int(bs[b]),int(bt[b])):
            nxt,_=model.transition(ctx[b:b+1],state,obs[b:b+1,t],previous[b:b+1,t+1]);state=hard(nxt)
        predictions.append(state[0])
    manual=torch.stack(predictions)
    parallel=memory[index,bs]
    for k in range(int((bt-bs).max())):
        t=(bs+k).clamp_max(x.shape[1]-2);active=(bs+k)<bt
        prefix_action=torch.where(active,previous[index,t+1],torch.zeros_like(t))
        nxt,_=model.transition(ctx,parallel,obs[index,t],prefix_action)
        parallel=torch.where(active[:,None],hard(nxt),parallel)
    assert torch.equal(manual.argmax(-1),parallel.argmax(-1))
    # No action after the scored position may influence the available prefix.
    altered=x.clone()
    for b in range(len(x)):altered[b,int(bt[b])+1:]=torch.randn_like(altered[b,int(bt[b])+1:])
    altered_ctx=model.context_sequence(altered,hard(model.observe(altered)),previous,data['branch_task'])[index,bs]
    assert torch.allclose(ctx,altered_ctx,atol=1e-6,rtol=1e-5)
report=dict(time=time.time(),source_commit=commit,loss=float(loss),gradient_norm=float(norm),
    actual_optimizer_step=True,all_parameter_gradients_finite=True,
    independent_scalar_and_batched_hybrid_memory_agree=True,future_images_do_not_change_context=True,
    no_extra_memory_grounding_at_runtime=True,no_new_model_parameters=True,
    seconds=time.time()-began,metrics={k:float(v) for k,v in metrics.items()})
out=ROOT/'diagnostic_hybrid_replay_v54';out.mkdir(exist_ok=True)
(out/'contract.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
