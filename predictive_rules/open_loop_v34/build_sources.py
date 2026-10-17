"""Prepare hard multi-step dynamics supervision without future observations.

This writes the next formal-training source. It launches no models or tests.
"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'action_grounded_v33'
BASE=HERE.parent/'raw_grounding_v31/hard_planning_objective.py'

def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError((before,text.count(before)))
    return text.replace(before,after)

text=BASE.read_text(encoding='utf-8')
text=text.replace('This module is prepared for a subsequent training revision; v31 is unchanged.',
    'Multi-step state supervision uses only the initial observed state as input.')
marker='\ndef planning_rule_objective(model,data,program):'
addition='''
def hard_open_loop_objective(model,program,data,probabilities,observed,horizon):
    """Predict hard symbolic futures without inserting any future observation.

    Grounded future IDs are detached supervision targets only. The recurrence
    sees an initial corrected integer ID, teacher actions, and generated T.
    Every subsequent premise is its own previous argmax prediction.
    """
    index=torch.arange(len(observed),device=observed.device)
    with torch.no_grad():
        initial=model.correction.selected(program['definitions'],program['context_raw'],
            observed[:,0],observed[:,0]).argmax(-1)
    state=F.one_hot(initial,probabilities.shape[-1]).float()
    valid=data['action_valid'][:,:horizon].float()
    losses=[];correct=[];predicted=[]
    with torch.autocast(device_type=probabilities.device.type,enabled=False):
        for t in range(horizon):
            actions=data['actions'][:,t]
            # This is exactly the conditional distribution of the selected
            # hard T row; only its argmax enters the next symbolic step.
            selected=probabilities[index,state.detach().argmax(-1),actions]
            # Zero forward value, derivative only to the previous hard state.
            candidate_rows=probabilities[index,:,actions].detach()
            through_state=torch.einsum('bs,bsk->bk',state-state.detach(),candidate_rows)
            following=selected+through_state
            target=observed[:,t+1]
            losses.append(-following.gather(-1,target[:,None]).squeeze(-1).clamp_min(1e-8).log())
            following_state=straight_through(following)
            ids=following_state.detach().argmax(-1)
            predicted.append(ids);correct.append(ids==target)
            state=torch.where(valid[:,t,None].bool(),following_state,state)
    normalizer=valid.sum().clamp_min(1)
    loss=(torch.stack(losses,1)*valid).sum()/normalizer
    accuracy=(torch.stack(correct,1).float()*valid).sum()/normalizer
    endpoint_index=data['goal_index'].clamp(1,horizon)-1
    endpoint=torch.stack(predicted,1)[index,endpoint_index]
    endpoint_target=observed[index,data['goal_index'].clamp_max(horizon)]
    return loss,dict(open_loop_dynamics_loss=float(loss.detach()),
        open_loop_transition_accuracy=float(accuracy.detach()),
        open_loop_teacher_endpoint_accuracy=float(endpoint.eq(endpoint_target).float().mean()),
        open_loop_examples=int(valid.sum()),open_loop_future_observation_corrections=0)

'''
text=replace_once(text,marker,addition+marker)
text=replace_once(text,
    "    return loss,dict(hard_rule_planning_loss=float(loss.detach()),hard_rule_planning_action_accuracy=float(accuracy.detach()),\n        hard_rule_planning_examples=int(valid.sum()),planning_gradients_to_rules=True,soft_runtime_states=False)",
    """    open_loop,open_metrics=hard_open_loop_objective(model,program,data,probabilities,observed,h)
    metrics=dict(hard_rule_planning_loss=float(loss.detach()),hard_rule_planning_action_accuracy=float(accuracy.detach()),
        hard_rule_planning_examples=int(valid.sum()),planning_gradients_to_rules=True,soft_runtime_states=False)
    metrics.update(open_metrics)
    # The caller weights this combined objective by0.25: backward-proof
    # imitation therefore remains0.25 and multi-step dynamics has weight0.5.
    return loss+2.*open_loop,metrics""")
(HERE/'hard_planning_objective.py').write_text(text,encoding='utf-8')

text=(PREVIOUS/'train.py').read_text(encoding='utf-8')
text=replace_once(text,"architecture='action_grounded_hard_rules_v33'","architecture='open_loop_hard_rules_v34'")
text=replace_once(text,'        no_additional_diagnostic_rollouts=True)',
    """        no_additional_diagnostic_rollouts=True,
        hard_open_loop_weight=.5,hard_open_loop_target='Detached grounded future integer IDs in the same generated definition dictionary',
        hard_open_loop_inputs='Initial corrected integer ID, generated T, recorded teacher actions; no future observation correction',
        teacher_path_exact_goal_legacy_metric='Endpoint with an observed correction at every teacher action; not an open-loop prediction score')""")
(HERE/'train.py').write_text(text,encoding='utf-8')
text=(PREVIOUS/'evaluate.py').read_text(encoding='utf-8')
text=replace_once(text,"=='action_grounded_hard_rules_v33'","=='open_loop_hard_rules_v34'")
(HERE/'evaluate.py').write_text(text,encoding='utf-8')
print('Prepared hard open-loop dynamics supervision; no execution performed.')
