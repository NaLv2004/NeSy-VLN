import copy
import pytest
from nesy_vla.reward_experiment import training_reward

OVERRIDES={'false_stop':-40.,'repeat_action':-.10,'repeat_action_growth':-.025,'repeat_action_max':-.25}


def reply(terms,occurrence=1,success=False):
    return {'reward':sum(terms.values()),'metrics':{'reward_terms':terms,'success':success,
        'action_repetition':{'occurrence':occurrence}},'done':'false_stop' in terms or success,
        'rgb':'same-pixels','feedback':{'collision':False}}


def test_success_geometry_actions_and_original_evidence_unchanged():
    r=reply({'success':25.,'time':-.04},success=True);before=copy.deepcopy(r)
    adjusted=training_reward(r,OVERRIDES)
    assert r==before and adjusted['reward']==r['reward']
    for key in ['metrics','done','rgb','feedback']:assert adjusted[key]==r[key]
    assert training_reward(r,{}) is r


def test_false_stop_cost_changes_only_training_reward():
    r=reply({'false_stop':-12.,'time':-.04})
    adjusted=training_reward(r,OVERRIDES)
    assert adjusted['reward']==-40.04 and adjusted['done']
    assert adjusted['metrics']['reward_terms']['false_stop']==-12.
    assert not adjusted['metrics']['success']


def test_all_recurrences_penalized_and_bounded_including_nonconsecutive():
    for n in range(2,33):
        r=reply({'time':-.04,'repeat_action':max(-2.,-.5-.25*(n-2))},n)
        r['metrics']['action_repetition']['consecutive_count']=1
        adjusted=training_reward(r,OVERRIDES)
        cost=adjusted['training_reward_audit']['training_terms']['repeat_action']
        assert -.25<=cost<=-.10
        assert cost==max(-.25,-.10-.025*(n-2))


@pytest.mark.parametrize('overrides',[{'success_distance_m':5.},{'false_stop':1.},
    {'repeat_action':0.},{**OVERRIDES,'repeat_action':0.},{'false_stop':float('nan')}])
def test_reject_changed_task_or_unpenalized_repeat(overrides):
    with pytest.raises(ValueError):training_reward(reply({'time':-.04}),overrides)
