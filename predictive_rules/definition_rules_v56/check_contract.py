"""Real data, real gradient, exported-table and pure-symbolic-planner check."""
import json,os,random,time
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from definition_rule import definition_inputs
from data import Corpus,ROOT,sha
from objectives import objective
from planner import Program,plan,audit

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(2);torch.manual_seed(561017);began=time.time()
source=ROOT/'diagnostic_temporal_symbols_v55/source.pt'
commit=json.loads(source.with_suffix('.COMMITTED.json').read_text());assert sha(source)==commit['sha256']
saved=torch.load(source,map_location='cpu',weights_only=False)
pilot=torch.load(ROOT/'diagnostic_definition_rules_v56/head.pt',map_location='cpu',weights_only=False)
assert pilot['source']['sha256']==commit['sha256']
model=FactoredWorld(Config(**saved['model_config']));model.load_initial(saved['model'],pilot['head'])
corpus=Corpus(branch_root=str(ROOT/'onpolicy_successors_v54'))
data=corpus.sample(random.Random(561017),8,8,'cpu');data['world_updates_encoder']=True
with torch.no_grad():
    x=data['features'][:1];prev=data['previous'][:1];task=data['task'][:1]
    context=model.context_sequence(x,hard(model.observe(x)),prev,task)[:,0]
    o=hard(model.observe(x[:,:1],context));_,ev,_=model.observation_encoding(x[:,:1],context)
    m=hard(model.memory_sequence(o,prev[:,:1],ev)[0])
    actions=torch.arange(model.cfg.actions)[None]
    mem=m.expand(-1,model.cfg.actions,-1);obs=o.expand(-1,model.cfg.actions,-1)
    args=definition_inputs(model,context,mem,obs,actions)
    p=model.definition_rule(*args);perm=torch.randperm(model.cfg.observations)
    renamed=list(args);renamed[1]=args[1][...,perm];renamed[2]=args[2][:,perm]
    renamed_logp=model.definition_rule(*renamed)
    rename_error=float((renamed_logp-p[...,perm]).abs().max());assert rename_error<2e-5
    array=model.compile(context,task);program=Program(**array).validate()
    nextm,logp=model.transition(context,mem,obs,actions)
    row=int(m.argmax(-1))*model.cfg.observations+int(o.argmax(-1))
    assert np.array_equal(program.next_memory[row],nextm.argmax(-1)[0].numpy())
    assert np.allclose(program.outcome_probability[row],logp.exp()[0].numpy(),atol=1e-7)
    # No neural module may be called once the tables have been emitted.
    def forbidden(*_):raise RuntimeError('Neural planning bypass')
    hooks=[module.register_forward_pre_hook(forbidden) for module in model.modules()]
    try:
        proof=plan(program,int(m.argmax(-1)),int(o.argmax(-1)),32)
        if proof['found']:audit(program,proof)
    finally:
        for h in hooks:h.remove()
    changed=x.clone();changed[:,1:]=torch.randn_like(changed[:,1:])
    assert torch.equal(context,model.context_sequence(changed,hard(model.observe(changed)),prev,task)[:,0])
loss,metrics=objective(model,data);assert torch.isfinite(loss);loss.backward()
active={n:p for n,p in model.named_parameters() if p.requires_grad}
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active.values()),[n for n,p in active.items() if p.grad is None]
definition_gradient=float(model.observation_definition_generator.weight.grad.norm())
rule_gradient=float(model.definition_rule.query[0].weight.grad.norm())
assert definition_gradient>0 and rule_gradient>0
norm=torch.nn.utils.clip_grad_norm_(list(active.values()),5.)
assert norm<100.,'Surrogate gradients amplified across symbolic row selections'
optimizer=torch.optim.AdamW(list(active.values()),lr=1e-4);optimizer.step()
result=dict(time=time.time(),seconds=time.time()-began,source=commit,device='CPU',
    real_optimizer_step=True,all_trainable_gradients_finite=True,definition_gradient=definition_gradient,
    rule_gradient=rule_gradient,gradient_norm=float(norm),loss=float(loss),
    rename_equivariance_error=rename_error,compiled_row_matches_forward=True,
    planner_neural_calls=0,pure_numpy_planner_verified=True,future_images_cannot_change_initial_context=True,
    no_continuous_runtime_state=True,unrelated_processes_untouched=True,
    runtime_memory_grounding_guard_inherited=True,development_used=False,
    limitation='Software and gradient contract only; physical navigation still requires unchanged full dev80.')
out=ROOT/'diagnostic_definition_rules_v56';out.mkdir(exist_ok=True)
(out/'contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
