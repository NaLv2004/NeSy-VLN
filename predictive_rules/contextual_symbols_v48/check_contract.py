"""Real training batch, causal symbols, exported proof and storage checks."""
import hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from data import ROOT,Corpus,sha
from objectives import objective
from planner import Program,plan,audit
from program_archive import snapshot,save_recipe,restore

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.2)
    torch.manual_seed(481017)
    out=ROOT/'diagnostic_contextual_symbols_v48';out.mkdir(exist_ok=True)
    source=ROOT/'diagnostic_sparse_support_v44';commit=json.loads((source/'source.COMMITTED.json').read_text())
    assert sha(source/'source.pt')==commit['sha256']
    saved=torch.load(source/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().train();model.load_v43(saved['model'])
    corpus=Corpus(branch_root=str(ROOT/'onpolicy_successors_v40'))
    d=corpus.sample(random.Random(481017),8,8,'cuda');d['world_updates_encoder']=True
    base=hard(model.observe(d['features']))
    contexts=model.context_sequence(d['features'],base,d['previous'],d['task'])
    index=torch.arange(8,device='cuda');ctx=contexts[index,d['start']]
    with torch.no_grad():
        old=model.observe(d['features']);new=model.observe(d['features'],ctx)
        initial_agreement=float((old.argmax(-1)==new.argmax(-1)).float().mean())
        assert initial_agreement==1.
        # Future images cannot affect the causal boundary generator.
        changed=d['features'].clone()
        for b in range(8):changed[b,int(d['start'][b])+1:]=torch.randn_like(changed[b,int(d['start'][b])+1:])
        other=model.context_sequence(changed,hard(model.observe(changed)),d['previous'],d['task'])[index,d['start']]
        assert torch.allclose(ctx,other,atol=1e-5,rtol=1e-5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    began=time.time();loss,metrics=objective(model,d);loss.backward()
    grad=model.observation_definition_generator.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    gradient=float(grad.norm());torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
    model.eval()
    with torch.no_grad():
        contexts=model.context_sequence(d['features'],hard(model.observe(d['features'])),d['previous'],d['task'])
        context=contexts[0:1,int(d['start'][0])];task=d['task'][0:1]
        local=model.observe(d['features'][0:1],context)
        # Distinct generated contexts must be able to define distinct symbols.
        delta=model.observation_definition_delta(context)
        delta2=model.observation_definition_delta(context+torch.randn_like(context))
        assert not torch.allclose(delta,delta2)
        assert local.shape==(1,34,64)
        program=Program(**model.compile(context,task)).validate()
        ob=int(local[0,int(d['start'][0])].argmax())
        proof=None
        for memory in range(32):
            candidate=plan(program,memory,ob,32)
            if candidate['found']:proof=candidate;break
        assert proof and audit(program,proof)
        archive=snapshot(model,out,0);(out/'case').mkdir(exist_ok=True)
        save_recipe(out/'case/recipe.npz',context,task,program,archive)
    # Archive restore exercises the saved standalone model, not live objects.
    restored=restore(out/'case/recipe.npz',device='cuda')
    # Restore returns a raw table mapping; compare using the actual API below.
    restored_table=restored[0] if isinstance(restored,tuple) else restored
    if isinstance(restored_table,dict):restored_table=Program(**restored_table).validate()
    assert restored_table.digest()==program.digest()
    report=dict(time=time.time(),source_commit=commit,initial_code_agreement=initial_agreement,
        future_images_cannot_affect_boundary_context=True,definition_generator_gradient=gradient,
        all_parameter_gradients_finite=True,generated_definition_shape=list(delta.shape),
        complete_program_shape=list(program.next_memory.shape),proof_audit_passed=True,archive_replay_exact=True,
        no_neural_action_head_added=True,loss=float(loss),seconds=time.time()-began,
        metrics={k:float(v) for k,v in metrics.items()})
    (out/'contract.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
