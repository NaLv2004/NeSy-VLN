"""Check train/runtime numerical agreement on identical training-only examples."""
import json,os,random
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.raw_boundary_world import RawBoundaryWorld,Config
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.data import sha

@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7';torch.set_num_threads(2)
    root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'diagnostics/categorical_precision_v20.json';assert not out.exists()
    source=root/'run_replayed_boundary_v20/checkpoint_000600.pt'
    assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False);model=RawBoundaryWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
    chosen=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
    features=RawBoundaryFeatures(root/'prequant_visual_features_v18','cuda');results={};summary={}
    modes=[('train_bf16_batch64',True,True,64),('runtime_tf32_batch64',False,True,64),('canonical_fp32_batch64',False,False,64),('runtime_tf32_batch1',False,True,1)]
    for name,amp,tf32,size in modes:
        torch.backends.cuda.matmul.allow_tf32=tf32;records=[]
        for begin in range(0,len(chosen),size):
            selected=chosen[begin:begin+size];d=batch(selected,'cuda');raw,_=features.demonstrations(selected)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                program=model.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0,boundary_features=raw)
                observed=model.training_targets(program,d['targets'],d['actions']).argmax(-1).squeeze(-1)
            rules=program['transitions'].argmax(-1);indices=torch.arange(len(selected),device='cuda')[:,None]
            one=rules[indices,observed[:,:-1],d['actions']]
            stop=rules[indices,observed[:,:-1],0]
            records.append(dict(observed=observed.cpu(),rules=rules.cpu(),goal=program['subgoal_logits'].argmax(-1).cpu(),
                one=one.cpu(),stop=stop.cpu(),valid=d['action_valid'].cpu(),state_valid=d['state_valid'].cpu()))
        r={k:torch.cat([v[k] for v in records]) for k in records[0]};results[name]=r;valid=r['valid'];sv=r['state_valid']
        summary[name]=dict(steps=int(valid.sum()),one_step_accuracy=float((r['one']==r['observed'][:,1:])[valid].float().mean()),
            STOP_identity=float((r['stop']==r['observed'][:,:-1])[valid].float().mean()),observed_unique=int(r['observed'][sv].unique().numel()))
    base=results['runtime_tf32_batch1'];comparisons={}
    for name,r in results.items():
        comparisons[name]=dict(observed_state_agreement=float((r['observed']==base['observed'])[base['state_valid']].float().mean()),
            initial_state_agreement=float((r['observed'][:,0]==base['observed'][:,0]).float().mean()),
            all_rule_argmax_agreement=float((r['rules']==base['rules']).float().mean()),goal_agreement=float((r['goal']==base['goal']).float().mean()))
    result=dict(training_only=True,checkpoint=str(source),sha256=sha(source),samples=len(chosen),mode_metrics=summary,agreement_with_runtime_tf32_batch1=comparisons)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
