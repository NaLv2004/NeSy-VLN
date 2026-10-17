"""One real CPU optimizer step; runtime isolation and categorical-input audit."""
import importlib.util,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from data import Corpus,ROOT,sha
from objectives import objective
from planner import Program,plan,audit

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(2);torch.manual_seed(551017);began=time.time()
source=ROOT/'onpolicy_successors_v54/source.pt'
commit=json.loads(source.with_suffix('.COMMITTED.json').read_text());assert sha(source)==commit['sha256']
saved=torch.load(source,map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config']));model.load_v54(saved['model'])
spec=importlib.util.spec_from_file_location('unchanged_v54_model',ROOT/'src_hybrid_replay_v54/model.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
baseline=old.FactoredWorld(old.Config(**saved['model_config'])).eval();baseline.load_state_dict(saved['model'])
corpus=Corpus(branch_root=str(ROOT/'onpolicy_successors_v54'))
data=corpus.sample(random.Random(551017),8,8,'cpu');data['world_updates_encoder']=True
with torch.no_grad():
    x=data['features'][:1];prev=data['previous'][:1];task=data['task'][:1]
    obs=hard(model.observe(x));context=model.context_sequence(x,obs,prev,task)[:,0]
    assert torch.equal(context,baseline.context_sequence(x,obs,prev,task)[:,0])
    a=baseline.compile(context,task);b=model.compile(context,task)
    assert a['goal']==b['goal'] and np.array_equal(a['next_memory'],b['next_memory']) and np.array_equal(a['outcome_probability'],b['outcome_probability'])
    def forbidden(*_):raise RuntimeError('Training-only head called by runtime symbolic derivation')
    hooks=[model.temporal_contrast.register_forward_pre_hook(forbidden),model.inverse.register_forward_pre_hook(forbidden)]
    program=Program(**model.compile(context,task)).validate();proof=plan(program,0,0,32)
    if proof['found']:audit(program,proof)
    for h in hooks:h.remove()
    altered=x.clone();altered[:,1:]=torch.randn_like(altered[:,1:])
    alternate=model.context_sequence(altered,hard(model.observe(altered)),prev,task)[:,0]
    assert torch.equal(context,alternate)
    # The same raw future features and program definitions reproduce exactly
    # the same IDs; no IDs are borrowed from another context.
    bc=context.expand(len(data['branch_negative_features']),-1)
    codes=hard(model.observe(data['branch_negative_features'],bc))
    assert torch.equal(codes.argmax(-1),model.observation_encoding(data['branch_negative_features'],bc)[0].argmax(-1))

inputs=[];outputs=[]
def capture(_,args,output):
    inputs.append(args[0].detach());outputs.append(output.squeeze(-1))
hook=model.temporal_contrast.register_forward_hook(capture)
loss,metrics=objective(model,data);hook.remove()
assert len(inputs)==2 and all(v.shape[-1]==model.cfg.memories+2*model.cfg.observations for v in inputs)
for values in inputs:
    assert torch.allclose(values,values.round(),atol=1e-6)
    for part in torch.split(values,[model.cfg.memories,model.cfg.observations,model.cfg.observations],-1):
        assert torch.allclose(part.sum(-1),torch.ones_like(part[...,0]),atol=1e-6)
ratio=.5*(F.binary_cross_entropy_with_logits(outputs[0],torch.zeros_like(outputs[0]))+
          F.binary_cross_entropy_with_logits(outputs[1],torch.ones_like(outputs[1])))
targets=[model.visual[1].weight,model.observation_definition_generator.weight,model.memory.weight_ih]
ratio_grads=torch.autograd.grad(ratio,targets,retain_graph=True)
assert all(torch.isfinite(g).all() and float(g.norm())>0 for g in ratio_grads)
before={n:p.detach().clone() for n,p in model.named_parameters()}
loss.backward();assert torch.isfinite(loss)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.step()
assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
result=dict(time=time.time(),seconds=time.time()-began,device='CPU',source_commit=commit,
    real_optimizer_step=True,all_parameter_gradients_finite=True,new_head_input_is_three_hard_ids=True,
    ratio_gradient_norms=dict(zip(['visual_encoder','observation_definitions','memory_encoder'],[float(g.norm()) for g in ratio_grads])),
    runtime_tables_exactly_preserved_before_training=True,runtime_training_heads_forbidden=True,
    future_images_cannot_change_initial_context=True,negative_images_use_current_program_definitions=True,
    loss=float(loss),gradient_norm=float(norm),metrics={k:float(v) for k,v in metrics.items()},
    note='Software/gradient contract on pinned precollection weights, not evidence of navigation improvement. Final initialization source is verified separately at launch.')
out=ROOT/'diagnostic_temporal_symbols_v55';out.mkdir(exist_ok=True)
(out/'contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
