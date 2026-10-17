"""Prepare one formal-training revision; no rollout or diagnostic execution."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'raw_grounding_v31'
PREVIOUS = HERE.parent / 'rule_planning_v32'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError((before, text.count(before)))
    return text.replace(before, after)


text = (RAW / 'raw_objectives.py').read_text(encoding='utf-8')
text = replace_once(text, '    inverse=equivalent_action_loss(logits,equivalent)', '''    inverse=equivalent_action_loss(logits,equivalent)
    # Ask the generated transition rules themselves which action explains each
    # actual successor ID. The independent inverse head cannot solve this loss
    # on behalf of the executable rules.
    log_rules=probabilities.clamp_min(1e-8).log()
    actual_after=z1.detach().argmax(-1)
    exact=log_rules.gather(2,actual_after[:,None].expand(-1,17,-1)).transpose(1,2)
    # Exact categorical lookup forward; this zero-valued term supplies the
    # representation gradient to the encoded successor. No soft runtime state.
    label_derivative=torch.einsum('btk,bak->bta',z1-z1.detach(),log_rules.detach())
    rule_inverse_scores=exact+label_derivative
    rule_inverse=equivalent_action_loss(rule_inverse_scores,equivalent)
    rule_inverse_match=equivalent.gather(-1,rule_inverse_scores.argmax(-1)[...,None]).float().mean()''')
text = replace_once(text, '    loss=dynamics+inverse+.125*anchor+.1*usage',
                    '    loss=dynamics+inverse+.5*rule_inverse+.125*anchor+.1*usage')
text = replace_once(text, "        branch_anchor=float(anchor.detach()),branch_loss=float(loss.detach()))", '''        branch_anchor=float(anchor.detach()),branch_loss=float(loss.detach()),
        branch_rule_inverse_loss=float(rule_inverse.detach()),
        branch_rule_inverse_accuracy=float(rule_inverse_match),
        branch_program_start=data['start'],branch_program_age=data['offset'],
        branch_observation_step=data['start']+data['offset'])''')
(HERE / 'raw_objectives.py').write_text(text, encoding='utf-8')

text = (RAW / 'raw_data.py').read_text(encoding='utf-8')
text = replace_once(text,
    "                    group=(t//10*10,t%10);self.groups[group].append((key,t));self.keys.append(group)",
    "                    self.groups[t].append((key,t));self.keys.append(t)")
text = replace_once(text,
    "        start,offset=rng.choice(self.keys);selected=rng.choices(self.groups[start,offset],k=size);rows=[]",
    """        # A runtime program may begin at any observation, so all-action
        # training must not assume the obsolete fixed ten-step boundaries.
        # The complete recorded prefix lets us choose a causal boundary without
        # rendering new data or changing the successor images.
        time_index=rng.choice(self.keys)
        start=rng.randint(0,time_index);offset=time_index-start
        selected=rng.choices(self.groups[time_index],k=size);rows=[]""")
(HERE / 'raw_data.py').write_text(text, encoding='utf-8')

text = (PREVIOUS / 'train.py').read_text(encoding='utf-8')
text = replace_once(text, "architecture='raw_hard_planning_v32'", "architecture='action_grounded_hard_rules_v33'")
text = replace_once(text,
    "hard_planning_forward='Same argmax successor and rounded confidence costs; zero-valued derivative surrogate only during training')",
    """hard_planning_forward='Same argmax successor and rounded confidence costs; zero-valued derivative surrogate only during training',
        rule_inverse_weight=.5,rule_inverse='Action likelihood obtained only from generated T and encoded hard successor IDs; identical observed outcomes form equivalence classes',
        counterfactual_program_boundaries='Uniform causal boundary0..t at each recorded training position t; no fixed ten-step clock',
        no_additional_diagnostic_rollouts=True)""")
(HERE / 'train.py').write_text(text, encoding='utf-8')
text = (PREVIOUS / 'evaluate.py').read_text(encoding='utf-8')
text = replace_once(text, "=='raw_hard_planning_v32'", "=='action_grounded_hard_rules_v33'")
(HERE / 'evaluate.py').write_text(text, encoding='utf-8')
print('Prepared action-grounded rules and causal arbitrary-boundary training.')
