"""Train future-memory rules while keeping the memory grounding alphabet fixed."""
import ast,shutil
from pathlib import Path
BASE=Path(__file__).parent
for name,gpus,visual in [('memory_dynamics_v77','2,3',False),('joint_memory_dynamics_v78','6,7',True)]:
    dst=BASE/name;dst.mkdir(exist_ok=True)
    for p in (BASE/'expanded_dynamics_v71').glob('*.py'):
        if p.name not in ('handoff.py','deploy.py','launch_handoff.py','collect_completion.py'):
            shutil.copy2(p,dst/p.name)
    p=dst/'base_objective.py';s=p.read_text()
    prefixes=('linear_visual_evidence.',) if visual else ()
    prefixes+=('definition_rule.','memory_predictor.','memory_rule.')
    s=s.replace("TRAINABLE=('linear_visual_evidence.','definition_rule.')",f'TRAINABLE={prefixes!r}')
    p.write_text(s)
    p=dst/'train_joint.py';s=p.read_text().replace('expanded_dynamics_v71',name)
    s=s.replace("=='6,7'",f"=='{gpus}'")
    old="old=ROOT/'diverse_history_v70';commit=json.loads((old/'source.COMMITTED.json').read_text());assert sha(old/'source.pt')==commit['sha256']\n    saved=torch.load(old/'source.pt',map_location='cpu',weights_only=False)"
    new="old=ROOT/'run_expanded_dynamics_v71';commit=json.loads((old/'best.COMMITTED.json').read_text());source=old/commit['filename'];assert sha(source)==commit['sha256']\n    saved=torch.load(source,map_location='cpu',weights_only=False)"
    assert old in s;s=s.replace(old,new)
    s=s.replace("dict(params=list(model.linear_visual_evidence.parameters()),lr=1e-4,weight_decay=.1)",
        "dict(params=[p for p in model.linear_visual_evidence.parameters() if p.requires_grad],lr=5e-5,weight_decay=.1)")
    s=s.replace("dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('linear_visual_evidence.')],lr=5e-5,weight_decay=1e-4)",
        "dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('definition_rule.')],lr=5e-5,weight_decay=1e-4),\n        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('memory_predictor.','memory_rule.'))],lr=2e-5,weight_decay=1e-4)")
    s=s.replace('learning_rates=dict(visual_linear=1e-4,rules=5e-5)',f'learning_rates=dict(visual_linear={5e-5 if visual else 0},observation_rules=5e-5,memory_rules=2e-5)')
    s=s.replace('frozen_context_definitions_goal_and_memory_predictor=True',
        f'frozen_context_definitions_goal_and_memory_grounding=True,visual_evidence_frozen={not visual},future_memory_predictor_trainable=True')
    s=s.replace('only generic visual residual and observation-transition decoder train','future-memory rule predictor and observation decoder train; actual-image memory encoder, state definitions and goals remain frozen')
    s=s.replace('torch.zeros(15,device=device','torch.zeros(17,device=device')
    marker="calibrated,_=objective.calibrated_goal(data.pairs[ids]);totals[13]+=calibrated.double()*n;totals[14]+=n"
    s=s.replace(marker,marker+"\n            totals[15]+=m['one_m_hits'];totals[16]+=m['first_m_nll']*n")
    s=s.replace('h1,c1,h2,c2,tnll,cb,cn=totals.cpu().tolist()','h1,c1,h2,c2,tnll,cb,cn,mhits,mnll=totals.cpu().tolist()')
    s=s.replace('symbolic_goal_brier=cb/cn)','symbolic_goal_brier=cb/cn,memory_accuracy=mhits/c1,memory_nll=mnll/c1)')
    s=s.replace("+4*row['symbolic_goal_brier']","+4*row['symbolic_goal_brier']+.5*row['memory_nll']")
    s=s.replace("qualifies=(measured['tpr']", "qualifies=(measured['memory_accuracy']>=baseline['memory_accuracy']-.02 and measured['tpr']")
    s=s.replace("ROOT/'run_goal_probability_v68/development_best/summary.json'", "ROOT/'run_expanded_dynamics_v71/development_best/summary.json'")
    # A real training update must send gradient into the future-memory predictor.
    s=s.replace("norm=torch.nn.utils.clip_grad_norm_", "memory_gradient=sum(float(p.grad.square().sum()) for n,p in model.named_parameters() if n.startswith(('memory_predictor.','memory_rule.')) and p.grad is not None)**.5\n        assert memory_gradient>0 and memory_gradient<float('inf'),memory_gradient\n        norm=torch.nn.utils.clip_grad_norm_")
    s=s.replace('gradient_norm=float(norm),peak_gib=', 'gradient_norm=float(norm),memory_gradient_norm=memory_gradient,peak_gib=')
    ast.parse(s);p.write_text(s)
    p=dst/'autobatch.py';s=p.read_text().replace('((16384,2048),(16384,4096),(32768,5120),(32768,6144))','((16384,1024),(16384,2048),(16384,4096),(32768,6144))');p.write_text(s)
    supervisor=(BASE/'stable_rules_v75/supervisor.py').read_text().replace('stable_rules_v75',name).replace("'2,3'",repr(gpus))
    (dst/'supervisor.py').write_text(supervisor)
    for p in dst.glob('*.py'):ast.parse(p.read_text())
    print(name,prefixes,gpus)
