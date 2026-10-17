"""Anonymous visual clustering only; labels never enter a deployed controller."""
import json,time
from collections import Counter,defaultdict
import torch
from torch.nn import functional as F

@torch.no_grad()
def probe(latents,keys,layouts,original,output):
    assert latents.shape[1:]==(18,256)
    held=torch.tensor([x in {112,113,120,121} for x in layouts],device=latents.device)
    assert held.any() and (~held).any()
    x=latents[~held].flatten(0,1);torch.manual_seed(2424)
    if len(x)>65536:x=x[torch.randperm(len(x),device=x.device)[:65536]]
    results={};began=time.time()
    def assignments(data,centers):
        return torch.cat([(z@centers.T).argmax(-1) for z in data.split(8192)])
    for size in [128,512]:
        centers=x[torch.randperm(len(x),device=x.device)[:size]].clone();history=[]
        for iteration in range(30):
            labels=assignments(x,centers);sums=torch.zeros_like(centers);sums.index_add_(0,labels,x)
            counts=torch.bincount(labels,minlength=size);empty=counts==0
            if empty.any():sums[empty]=x[torch.randperm(len(x),device=x.device)[:int(empty.sum())]]
            centers=F.normalize(sums,dim=-1)
            history.append(float((x*centers[labels]).sum(-1).mean()))
            if (iteration+1)%10==0:print(json.dumps(dict(vocabulary=size,iteration=iteration+1,mean_cosine=history[-1],seconds=time.time()-began)),flush=True)
        labels=assignments(latents.flatten(0,1),centers).reshape(-1,18).cpu().tolist()
        groups={}
        for name in ['fitting_layouts','excluded_training_layouts','all']:
            rules=defaultdict(Counter);used=Counter();branches=0;programs=set();alias=0
            for key,layout,row in zip(keys,layouts,labels):
                is_held=layout in {112,113,120,121}
                if name!='all' and is_held!=(name=='excluded_training_layouts'):continue
                programs.add(key);used.update(row)
                for action,target in enumerate(row[1:]):rules[key,row[0],action][target]+=1;branches+=1
                alias+=17-len(set(row[1:]))
            correct=sum(max(counts.values()) for counts in rules.values())
            conflicting=[counts for counts in rules.values() if len(counts)>1]
            groups[name]=dict(branches=branches,programs=len(programs),unique_state_ids=len(used),
                rule_rows=len(rules),conflicting_rule_rows=len(conflicting),
                deterministic_label_fit_ceiling=correct/branches,
                branches_in_contradictory_rows=sum(sum(x.values()) for x in conflicting),
                distinct_successor_states_per_current=(branches-alias)/(branches/17))
        results[str(size)]=dict(groups=groups,fitting_frames=len(x),mean_cosine_by_iteration=history)
        torch.save(dict(centers=centers.cpu(),fitting_layouts=sorted(set(layouts)-{112,113,120,121}),
            excluded_training_layouts=[112,113,120,121],input='normalized frozen visual/history feature, no named semantic targets',
            source_checkpoint=original['checkpoint'],source_sha256=original['sha256'],not_deployed=True),
            output.with_name(f'state_resolution_v24_centers_{size}.pt'))
    report=dict(training_only=True,diagnostic_not_deployed=True,no_optimizer_step=True,no_new_rollouts=True,
        source_checkpoint=original['checkpoint'],source_sha256=original['sha256'],
        feature_shape=list(latents.shape),centers_fit_without_actions_goals_or_semantic_labels=True,
        excluded_training_layouts=[112,113,120,121],development_and_final_validation_unused=True,
        current128_label_fit_ceiling=original['best_possible_deterministic_accuracy_on_recorded_states'],
        current_dynamics_accuracy=original['current_mean_accuracy'],vocabularies=results,seconds=time.time()-began)
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
