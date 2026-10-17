"""Training-only reward accounting; never changes actions or success criteria."""
import math

ALLOWED={'false_stop','repeat_action','repeat_action_growth','repeat_action_max'}


def training_reward(response,overrides=None):
    if not overrides:return response
    if set(overrides)-ALLOWED:raise ValueError('Only false-STOP and repeat costs may be overridden')
    if any(not math.isfinite(float(v)) for v in overrides.values()):raise ValueError('Non-finite reward cost')
    if 'false_stop' in overrides and overrides['false_stop']>=0:raise ValueError('Failed STOP must remain penalized')
    repeat_keys=ALLOWED-{'false_stop'}
    if repeat_keys&overrides.keys():
        if not repeat_keys<=overrides.keys():raise ValueError('Specify the complete repeated-action cost')
        if not overrides['repeat_action_max']<=overrides['repeat_action']<0 or overrides['repeat_action_growth']>0:
            raise ValueError('Every repeated action must retain a bounded negative penalty')
    metrics=response['metrics'];original=dict(metrics['reward_terms']);terms=dict(original)
    if abs(sum(original.values())-response['reward'])>1e-6:raise ValueError('Reward terms do not match the environment')
    if 'false_stop' in terms and 'false_stop' in overrides:terms['false_stop']=overrides['false_stop']
    if 'repeat_action' in terms and repeat_keys<=overrides.keys():
        occurrence=metrics['action_repetition']['occurrence']
        if occurrence<2:raise ValueError('Repeated-action metadata inconsistent')
        terms['repeat_action']=max(overrides['repeat_action_max'],
            overrides['repeat_action']+overrides['repeat_action_growth']*(occurrence-2))
    # Preserve the original environment metrics and terminal condition. The
    # allowlisted public observation/feedback path never receives these fields.
    return dict(response,reward=sum(terms.values()),training_reward_audit={
        'original_reward':response['reward'],'original_terms':original,'training_terms':terms})
