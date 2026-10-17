import copy,itertools,json
import pytest,torch
from nesy_vla.small_generator import SymbolicProgramGenerator
from nesy_vla.conjunctive_logic import ConjunctiveRuleLogic
from nesy_vla.program import Program
from nesy_vla.symbols import tokenize,DiscreteState
from nesy_vla.logic import NoSupportedAction


def fixture():
    torch.set_num_threads(2);torch.manual_seed(3701)
    generator=SymbolicProgramGenerator(dict(k=4,d=32,r=12,a=3,structured=True,
        multi_effects=True,action_semantics='conjunctive_rules',neural_conjunction_gates=True),32)
    p=generator('automatically encoded task',[],[])
    logic=ConjunctiveRuleLogic(4,3,12,neural_action_gates=True)
    return generator,p,logic


def test_identity_initialization_and_legacy_default():
    _,p,logic=fixture();p.validate()
    states=tokenize(torch.tensor(list(itertools.product([0.,1.],repeat=4))))
    enabled=logic.probabilities_for_states(p,states)
    old=p.map(lambda t:t);old.neural_conjunction_gates=False
    assert torch.equal(enabled,logic.probabilities_for_states(old,states))
    record=p.record();json.dumps(record,allow_nan=False)
    restored=Program.from_record(record);restored.validate()
    assert restored.neural_conjunction_gates
    assert torch.equal(enabled,logic.probabilities_for_states(restored,states))
    record.pop('neural_conjunction_gates')
    assert not Program.from_record(record).neural_conjunction_gates


def test_gates_learn_but_cannot_override_false_premises():
    _,p,logic=fixture();state=tokenize(torch.tensor([1.,0.,1.,0.]))
    proof=logic.infer(p,state);(-proof['probs'][1].log()).backward()
    assert logic.action_rule_gate[-1].weight.grad.abs().sum()>0
    with torch.no_grad():logic.action_rule_gate[-1].bias.fill_(100.)
    hard=p.harden();hard.conjunctive_conditions[:]=torch.tensor([0.,0.,1.])
    with pytest.raises(NoSupportedAction):logic.infer(hard,tokenize(torch.ones(4)))


def test_generated_conclusions_are_the_only_action_source():
    _,p,logic=fixture()
    with torch.no_grad():
        torch.nn.init.normal_(logic.action_rule_gate[-1].weight,std=.5)
    for action in range(3):
        changed=p.map(lambda t:t.clone())
        changed.action_heads[logic.action_slice]=torch.nn.functional.one_hot(torch.tensor(action),3).float()
        proof=logic.infer(changed,tokenize(torch.tensor([1.,0.,1.,0.])))
        torch.testing.assert_close(proof['probs'],torch.nn.functional.one_hot(torch.tensor(action),3).float())
        torch.testing.assert_close(proof['probs'],proof['action_contribution'].sum(0))
    changed.weights[logic.action_slice]=0
    with pytest.raises(NoSupportedAction):logic.infer(changed,tokenize(torch.ones(4)))


def test_gate_state_boundary_and_exact_model_replay():
    _,p,logic=fixture()
    with torch.no_grad():torch.nn.init.normal_(logic.action_rule_gate[-1].weight,std=.5)
    with pytest.raises(TypeError):logic.action_log_gates(p,torch.rand(2,4))
    with pytest.raises(ValueError):logic.action_log_gates(p,DiscreteState(torch.ones(2,4,dtype=torch.long),torch.rand(2,4)))
    restored=ConjunctiveRuleLogic(4,3,12,neural_action_gates=True)
    restored.load_state_dict(copy.deepcopy(logic.state_dict()))
    state=tokenize(torch.tensor([1.,0.,1.,0.]))
    assert torch.equal(logic.infer(p,state)['probs'],restored.infer(Program.from_record(p.record()),state)['probs'])
