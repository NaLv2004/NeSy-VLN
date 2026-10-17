import importlib.util,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from model import FactoredWorld,Config,hard
from data import ROOT,Corpus,sha
from objectives import objective
from planner import Program,plan,audit
from program_archive import snapshot,save_recipe,restore

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.2);torch.manual_seed(521017)
    out=ROOT/'diagnostic_goal_aware_symbols_v52';source=out/'source.pt'
    commit=json.loads(source.with_suffix('.COMMITTED.json').read_text());assert sha(source)==commit['sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().train();model.load_v48(saved['model'])
    spec=importlib.util.spec_from_file_location('original_v48',ROOT/'src_contextual_symbols_v48/model.py')
    old_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old_module;spec.loader.exec_module(old_module)
    old=old_module.FactoredWorld(old_module.Config(**saved['model_config'])).cuda().eval();old.load_state_dict(saved['model'])
    corpus=Corpus(branch_root=str(ROOT/'onpolicy_successors_v40'))
    data=corpus.sample(random.Random(521017),8,8,'cuda');data['world_updates_encoder']=True
    index=torch.arange(8,device='cuda');x=data['features'];task=data['task']
    with torch.no_grad():
        global_o=hard(model.observe(x));contexts=model.context_sequence(x,global_o,data['previous'],task);context=contexts[index,data['start']]
        old_context=old.context_sequence(x,global_o,data['previous'],task)[index,data['start']]
        assert torch.equal(context,old_context)
        assert torch.equal(model.observe(x,context),old.observe(x,old_context))
        future=x.clone()
        for b in range(8):future[b,int(data['start'][b])+1:]=torch.randn_like(future[b,int(data['start'][b])+1:])
        changed=model.context_sequence(future,hard(model.observe(future)),data['previous'],task)[index,data['start']]
        assert torch.allclose(context,changed,atol=1e-5,rtol=1e-5)
    del old
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    began=time.time();loss,metrics=objective(model,data);loss.backward()
    g=model.abstraction_task.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.norm()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    gradient=float(g.norm());torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();model.eval()
    with torch.no_grad():
        global_o=hard(model.observe(x))
        a=model.context_sequence(x,global_o,data['previous'],torch.zeros_like(task))[index,data['start']]
        b=model.context_sequence(x,global_o,data['previous'],torch.ones_like(task))[index,data['start']]
        assert not torch.allclose(a,b,atol=1e-8,rtol=0)
        definition_difference=float((model.observation_definition_delta(a)-model.observation_definition_delta(b)).norm())
        assert definition_difference>0
        # Task indices carry no hard-coded meaning. Permuting learned task
        # embedding rows and task IDs together leaves the program unchanged.
        task_weights=model.task.weight.clone();basis_weights=model.abstraction_task.weight.clone()
        original_context=model.context_sequence(x,global_o,data['previous'],task)[index,data['start']]
        original_goal=model.goal_logits(original_context,task)
        model.task.weight.copy_(task_weights.flip(0));model.abstraction_task.weight.copy_(basis_weights.flip(0))
        relabeled_context=model.context_sequence(x,global_o,data['previous'],1-task)[index,data['start']]
        assert torch.equal(original_context,relabeled_context)
        for u,v in zip(original_goal,model.goal_logits(relabeled_context,1-task)):assert torch.equal(u,v)
        model.task.weight.copy_(task_weights);model.abstraction_task.weight.copy_(basis_weights)
        context=original_context[:1];task=task[:1]
        program=Program(**model.compile(context,task)).validate()
        local=model.observe(x[:1],context);observed=int(local[0,int(data['start'][0])].argmax())
        proofs=[plan(program,m,observed,32) for m in range(model.cfg.memories)]
        proof=next(p for p in proofs if p['found']);assert audit(program,proof)
        archive=snapshot(model,out,0);(out/'case').mkdir(exist_ok=True)
        save_recipe(out/'case/recipe.npz',context,task,program,archive)
    restored=restore(out/'case/recipe.npz',device='cuda');assert restored.digest()==program.digest()
    report=dict(time=time.time(),source_commit=commit,seconds=time.time()-began,loss=float(loss),
        initial_v48_context_and_codes_exact=True,future_images_cannot_affect_boundary_context=True,
        task_embedding_gradient=gradient,task_conditioned_definition_difference=definition_difference,
        all_parameter_gradients_finite=True,task_index_permutation_equivariant=True,
        complete_program_shape=list(program.next_memory.shape),proof_audit_passed=True,
        archive_replay_exact=True,no_neural_action_head_added=True,
        metrics={k:float(v) for k,v in metrics.items()})
    (out/'contract.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
