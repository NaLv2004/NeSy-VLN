"""Execution audits of the actual logic bottleneck, never a policy input."""
import torch
from .logic import noisy_or


@torch.no_grad()
def audit_policies(model,policies):
    count=changed=collapsed=programs=0;cosines=[];observed=[];differences=[];codes=[];errors=[]
    for policy in policies:
        for chunk in policy.chunks:
            p=chunk['program'];definitions=torch.nn.functional.normalize(p.definitions,dim=-1)
            gram=definitions@definitions.T;mask=~torch.eye(len(gram),device=gram.device,dtype=torch.bool)
            cosines.append(float(gram[mask].abs().mean()))
            heads=p.action_heads[model.logic.action_slice].argmax(-1)
            collapsed+=int((heads==heads[0]).all());programs+=1
            for row in chunk['rows']:
                b=p.definitions.new_tensor(row['belief']);c=p.definitions.new_tensor(row['confidence'])
                def choose(state):
                    s=model.logic.fire(p,state,c,model.logic.action_slice)
                    q=noisy_or(s[:,None]*p.action_heads[model.logic.action_slice],0)
                    return q/q.sum()
                if model.cfg.get('discrete_reasoning',False):
                    from .symbols import tokenize
                    bits=p.definitions.new_tensor(row['symbol_codes']);codes.append(bits)
                    original=model.logic.infer(p,tokenize(bits))['probs']
                    flipped=model.logic.infer(p,tokenize(1-bits))['probs']
                    saved=p.definitions.new_tensor(row['support']);saved=saved/saved.sum()
                    errors.append(float((original-saved).abs().max()))
                else:original=choose(b);flipped=choose(1-b)
                changed+=int(original.argmax()!=flipped.argmax());count+=1
                differences.append(float((original-flipped).abs().sum()/2))
                observed.append(p.definitions.new_tensor(row['observed_truth']))
    answer={'steps':count,'belief_flip_changed_greedy':changed,
            'belief_flip_changed_fraction':changed/max(1,count),
            'mean_belief_flip_probability_tv':sum(differences)/max(1,len(differences)),
            'programs':programs,'all_action_rules_same_argmax':collapsed,
            'mean_absolute_definition_cosine':sum(cosines)/max(1,len(cosines)),
            'observed_temporal_std':float(torch.stack(observed).std(0,unbiased=False).mean()) if observed else 0.}
    if codes:
        values=torch.stack(codes);frequencies=values.mean(0)
        answer.update(discrete_states=len(torch.unique(values,dim=0)),
            variable_predicates=int(((frequencies>0)&(frequencies<1)).sum()),
            replay_probability_max_error=max(errors),discrete_boundary=True)
    return answer
