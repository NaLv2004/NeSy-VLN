"""Read-only gradient attribution on training data, without an optimizer step."""
import inspect,json,os,random
from pathlib import Path
import torch
from predictive_nesy.raw_boundary_world import RawBoundaryWorld,Config
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.raw_visual_objectives import objective
from predictive_nesy.data import sha

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7';torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'diagnostics/rule_gradient_conflict_v21.json';assert not out.exists()
    source=root/'run_replayed_boundary_v20/checkpoint_000600.pt'
    assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False);model=RawBoundaryWorld(Config(**saved['model_config'])).cuda().train();model.load_state_dict(saved['model'])
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('definition_rules.','subgoal.','raw_')))
    # Expose the existing objective's intermediate tensors; no formula changes.
    code=inspect.getsource(objective);assert code.count('return loss,metrics')==1
    namespace=dict(objective.__globals__);exec(compile(code.replace('return loss,metrics','return loss,metrics,losses'),'<objective-attribution>','exec'),namespace)
    attributed=namespace['objective'];features=RawBoundaryFeatures(root/'prequant_visual_features_v18','cuda')
    windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts']);reports=[]
    parameters=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('definition_rules.','raw_'))]
    def vector(loss):
        gs=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten() for p,g in zip(parameters,gs)])
    for start in [0,10]:
        choices=[w for w in windows if w['start']==start and (w['supervised']|~w['action_valid']).all() and (start or w['goal_index']>=4)]
        for seed in [744,3919]:
            chosen=random.Random(seed).choices(choices,k=128);d=batch(chosen,'cuda');raw,_=features.demonstrations(chosen);width=min(start+1,11)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=model.generate(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start,boundary_features=raw)
                loss,metrics,parts=attributed(model,d,proof_from_predicted_state=True,program=p)
            dynamics=2*parts['one_step_dynamics']+.5*parts['dynamics']+.5*parts['teacher_path_goal']+.3*parts['rule_inverse']
            gv=vector(dynamics);gp=vector(parts['proof_action_imitation']);gg=vector(parts['subgoal_prediction'])
            nv=float(gv.norm());np_=float(gp.norm());ng=float(gg.norm());dot=float(gv@gp);goal_dot=float(gv@gg)
            reports.append(dict(start=start,seed=seed,batch=128,world_gradient_norm=nv,proof_gradient_norm=np_,goal_gradient_norm=ng,
                proof_world_cosine=dot/max(nv*np_,1e-20),goal_world_cosine=goal_dot/max(nv*ng,1e-20),
                proof_step_increases_world_loss_first_order=dot<0,goal_step_increases_world_loss_first_order=goal_dot<0,
                proof_to_world_norm_ratio=np_/max(nv,1e-20),metrics=metrics))
            del p,loss,parts,gv,gp,gg,dynamics,d,raw
    result=dict(checkpoint=str(source),sha256=sha(source),training_only=True,no_optimizer_step=True,
        scope='Joint generated-rule and shared raw-context parameters; exact existing supervised losses',reports=reports)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
