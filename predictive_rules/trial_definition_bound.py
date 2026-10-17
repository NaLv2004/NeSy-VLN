"""Fit definition-conditioned rules on a frozen, learned visual alphabet."""
import copy,hashlib,json,os,random,time
from pathlib import Path
import torch
from predictive_nesy.definition_bound_model import DefinitionBoundModel,Config
from predictive_nesy.visual_memory_subgoal_training import objective,transition_objective
from predictive_nesy.subgoal_data import load_windows,batch,TransitionBank

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'definition_bound_trial'
if (out/'launcher.json').exists():raise RuntimeError('Bounded trial was already launched; do not restart')
out.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
torch.manual_seed(815);rng=random.Random(815)
source=root/'run_visual_graph_subgoal_v5/checkpoint_004000.pt'
saved=torch.load(source,map_location='cpu',weights_only=False)
model=DefinitionBoundModel(Config(**saved['model_config'])).cuda()
transfer={k:v for k,v in saved['model'].items() if not k.startswith(('transitions.','subgoal.'))}
report=model.load_state_dict(transfer,strict=False)
assert not report.unexpected_keys and all(k.startswith(('definition_rules.','visual_memory.','memory_projection.')) for k in report.missing_keys)
for name,p in model.named_parameters():p.requires_grad_(name.startswith('definition_rules.'))
fixed={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('definition_rules.')}
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
starts=[w['start'] for w in windows];groups={s:[w for w in windows if w['start']==s] for s in set(starts)}
banks=[TransitionBank(root/name,saved['manifest']['data']['train_layouts']) for name in
       ['counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5']]
optimizer=torch.optim.AdamW(model.definition_rules.parameters(),lr=3e-4,weight_decay=1e-4)
manifest=copy.deepcopy(saved['manifest']);manifest.update(architecture='anonymous_definition_bound_subgoal_v7',
    rule_generation='all rows condition on generated definitions and causal context; renaming-equivariant',
    subgoal_generation='query over generated definitions; unmodified categorical argmax',
    definition_bound_intervention=dict(source=str(source),updates=1000,learning_rate=3e-4,trainable=['definition_rules'],
        definition_rule_heads_initialized_fresh=True,all_grounding_and_context_fixed=True,training_only=True))
(out/'launcher.json').write_text(json.dumps(dict(pid=os.getpid(),gpu=7,time=time.time(),source=str(Path(__file__).parent))))
begin=time.time()
for step in range(1,1001):
    optimizer.zero_grad(set_to_none=True);model.train();start=rng.choice(starts)
    data=batch(rng.choices(groups[start],k=256),'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,data)
    loss.backward();del loss,data
    data=rng.choice(banks).sample(rng,16,'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,extra=transition_objective(model,data)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.definition_rules.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite rule gradient')
    optimizer.step();del loss,data
    if step%100==0:
        torch.cuda.synchronize();metrics.update(extra,step=step,seconds=time.time()-begin,gradient_norm=float(norm),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
assert all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in fixed.items())
manifest['definition_bound_intervention']['frozen_grounding_bitwise_unchanged']=True
torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_config=model.export_config(),
    manifest=manifest,update=5000),out/'candidate.tmp');os.replace(out/'candidate.tmp',out/'candidate.pt')
commit=dict(path='candidate.pt',sha256=hashlib.sha256((out/'candidate.pt').read_bytes()).hexdigest(),update=5000,time=time.time())
(out/'candidate.COMMITTED.json').write_text(json.dumps(commit,indent=2))
print(json.dumps(dict(completed=True,seconds=time.time()-begin,checkpoint=commit)),flush=True)
