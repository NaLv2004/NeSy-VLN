"""Numerical, gradient and execution-contract checks for512-state training."""
import json,os,random,time
from pathlib import Path
import torch
from predictive_nesy.state512_rule_world import State512RuleWorld,Config,export_program
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank
from predictive_nesy.selected_row_objectives import demonstration_objective,transition_objective
from predictive_nesy.separated_goal_objectives import objective as full_demonstration_objective
from predictive_nesy.separated_transition_objectives import transition_objective as full_transition_objective
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.data import sha
from train_state512_v25 import TrainingModule

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7';torch.set_num_threads(2);torch.manual_seed(2526);rng=random.Random(2526)
    root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'preflight_state512_v25';assert not out.exists()
    path=root/'initialization_state512_v25r1/initial.pt';assert sha(path)==json.loads(path.with_suffix('.COMMITTED.json').read_text())['sha256']
    assert json.loads((path.parent/'report.json').read_text())['passed']
    saved=torch.load(path,map_location='cpu',weights_only=False);model=State512RuleWorld(Config(**saved['model_config'])).cuda().train();model.load_state_dict(saved['model'])
    prefixes=('definition_rules.','raw_','subgoal.','goal_','rule_evidence.')
    for n,p in model.named_parameters():p.requires_grad_(n.startswith(prefixes))
    frozen={n:v.cpu().clone() for n,v in model.state_dict().items() if not n.startswith(prefixes)}
    params=[p for p in model.parameters() if p.requires_grad]
    world=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('definition_rules.','raw_','rule_evidence.'))]
    goals=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('subgoal.','goal_'))]
    features=ReplayedBoundaryFeatures(root/'prequant_visual_features_v18',root/'replayed_boundary_features_v20','cuda')
    windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
    selected=rng.choices([w for w in windows if w['start']==10],k=256)
    data=batch(selected[:3],'cuda');raw,_=features.demonstrations(selected[:3])
    def prepare(d,r):return model.prepare(d['task'],d['history'],d['previous_actions'],d['history_length'],start=10,boundary_features=r)
    def full(p):return {**{k:v for k,v in p.items() if not k.startswith('rule_')},'transitions':model.rows(p)}
    def vector(loss):
        gradients=torch.autograd.grad(loss,params,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().float() for p,g in zip(params,gradients)])
    torch.backends.cuda.matmul.allow_tf32=False
    p=prepare(data,raw);fp=full(p);old_loss,old_metrics=full_demonstration_objective(model,data,proof_from_predicted_state=True,program=fp)
    old_value=float(old_loss.detach());old_gradient=vector(old_loss);del p,fp,old_loss
    p=prepare(data,raw);new_loss,new_metrics=demonstration_objective(model,data,p);new_value=float(new_loss.detach());new_gradient=vector(new_loss)
    relative=float((new_gradient-old_gradient).norm()/old_gradient.norm().clamp_min(1e-12));absolute=float((new_gradient-old_gradient).abs().max())
    assert abs(old_value-new_value)<1e-4 and relative<1e-3,(old_value,new_value,relative,absolute)
    equivalence=dict(full_loss=old_value,selected_row_loss=new_value,relative_gradient_error=relative,max_gradient_error=absolute)
    del p,new_loss,old_gradient,new_gradient
    p=prepare(data,raw)
    ids=torch.tensor([[0,128,511],[127,260,302],[80,405,510]],device='cuda')
    all_rows=model.rows(p);selected_rows=model.rows(p,ids)
    reference=all_rows.gather(1,ids[:,:,None,None].expand(-1,-1,17,512))
    error=float((selected_rows-reference).abs().max())
    row_argmax_identical=bool(torch.equal(selected_rows.argmax(-1),reference.argmax(-1)))
    # Attention reductions can differ by a few FP32 ulps with a shorter query
    # axis. Require both a tight probability bound and identical discrete rules.
    assert error<1e-5 and row_argmax_identical,(error,row_argmax_identical)
    del p,all_rows,selected_rows
    p=prepare(data,raw);loss,_,parts=demonstration_objective(model,data,p,return_parts=True)
    def mass(value,parameters):
        gs=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
        return sum(float(g.abs().sum()) for g in gs if g is not None)
    gl=parts['subgoal_prediction']+parts['proof_action_imitation'];wl=2*parts['one_step_dynamics']
    routing=dict(goal_loss_to_world=mass(gl,world),world_loss_to_goal=mass(wl,goals),goal_loss_to_goal=mass(gl,goals),world_loss_to_world=mass(wl,world))
    assert routing['goal_loss_to_world']==routing['world_loss_to_goal']==0 and routing['goal_loss_to_goal']>0 and routing['world_loss_to_world']>0
    del p,loss,parts,gl,wl
    torch.backends.cuda.matmul.allow_tf32=True
    with torch.no_grad():
        p=prepare(data,raw);labels=model.training_targets(p,data['targets'],data['actions']).argmax(-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            ap=prepare(data,raw);amp_labels=model.training_targets(ap,data['targets'],data['actions']).argmax(-1)
        assert torch.equal(labels,amp_labels)
    del p,ap
    bank=TransitionBank(saved['manifest']['transition_banks'][0]['path'],saved['manifest']['data']['train_layouts'])
    d=bank.sample(rng,3,'cuda');rr,avail=features.transitions(d,bank.directory);width=d['history'].shape[1]
    def transition_program():return model.prepare(d['task'],d['history'],d['previous_actions'],torch.full((3,),width,device='cuda',dtype=torch.long),start=d['start'],boundary_features=rr)
    torch.backends.cuda.matmul.allow_tf32=False
    p=transition_program();old_loss,_=full_transition_objective(model,d,program=full(p));old_value=float(old_loss.detach());old_gradient=vector(old_loss);del p,old_loss
    p=transition_program();new_loss,_=transition_objective(model,d,p);new_value=float(new_loss.detach());new_gradient=vector(new_loss)
    trel=float((new_gradient-old_gradient).norm()/old_gradient.norm().clamp_min(1e-12))
    assert abs(old_value-new_value)<1e-4 and trel<1e-3,(old_value,new_value,trel)
    transition_equivalence=dict(full_loss=old_value,selected_row_loss=new_value,relative_gradient_error=trel)
    del p,d,rr,old_gradient,new_gradient,new_loss
    torch.backends.cuda.matmul.allow_tf32=True
    model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4);wrapper=TrainingModule(model)
    d=batch(selected,'cuda');d['boundary_features'],d['boundary_available']=features.demonstrations(selected);began=time.time()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=wrapper(d,'demonstrations')
    loss.backward();del d,loss
    d=bank.sample(rng,128,'cuda');d['boundary_features'],d['boundary_available']=features.transitions(d,bank.directory)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=wrapper(d,'transitions')
    loss.backward();del d,loss
    for group in [world,goals]:
        norm=torch.nn.utils.clip_grad_norm_(group,1.);assert torch.isfinite(norm)
    opt.step();torch.cuda.synchronize();elapsed=time.time()-began;peak=torch.cuda.max_memory_allocated()/2**30
    assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items())
    model.eval();lengths=torch.full_like(data['history_length'],5);hc=data['history'].clone();ac=data['previous_actions'].clone();rc=raw.clone()
    hc[:,5:]=torch.randint(64,hc[:,5:].shape,device='cuda');ac[:,5:]=torch.randint(18,ac[:,5:].shape,device='cuda');rc[:,5:]=torch.randn_like(rc[:,5:])*100
    gate=NeuralBoundaryGate([model])
    with torch.no_grad(),gate.boundary(10):
        a=model.generate(data['task'],data['history'],data['previous_actions'],lengths,start=10,boundary_features=raw)
        b=model.generate(data['task'],hc,ac,lengths,start=10,boundary_features=rc)
        causal={k:bool(torch.equal(a[k],b[k])) for k in a if isinstance(a[k],torch.Tensor)}
        program=export_program(a);assert program.transitions.shape==(512,17,512)
    assert all(causal.values()),causal
    rejected=False
    try:model.generate(data['task'],data['history'],data['previous_actions'],lengths,start=10,boundary_features=raw)
    except RuntimeError as exc:rejected='between rule revisions' in str(exc)
    finally:gate.close()
    assert rejected
    report=dict(passed=True,source_sha256=sha(path),states=512,complete_export_shape=list(program.transitions.shape),
        demonstration_equivalence=equivalence,transition_equivalence=transition_equivalence,selected_row_max_error=error,selected_row_argmax_identical=row_argmax_identical,
        gradient_routing=routing,AMP_and_runtime_grounding_identical=True,causal_future_mutation_invariance=causal,
        between_revision_neural_call_rejected=True,frozen_tensors_unchanged=len(frozen),real_update_seconds=elapsed,memory_peak_gib=peak,
        demo_batch=256,transition_batch=128,metrics=metrics,transition_metrics=extra,time=time.time())
    out.mkdir();(out/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
