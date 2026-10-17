import itertools,unittest
import torch
from nesy_vla.small_generator import SymbolicProgramGenerator
from nesy_vla.branch_logic import TotalBranchLogic
from nesy_vla.symbols import tokenize
from nesy_vla.program import Program,describe
from nesy_vla.logic import NoSupportedAction


class BranchLogicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(33)
        self.cfg={'k':4,'d':32,'r':12,'a':3,'structured':True,'multi_effects':True,'action_semantics':'branch_tree'}
        self.generator=SymbolicProgramGenerator(self.cfg,32)
        self.program=self.generator('automatically encoded task',[],[])
        self.logic=TotalBranchLogic(4,3,12)
    def test_all_boolean_states_have_unit_proof_mass(self):
        for bits in itertools.product([0.,1.],repeat=4):
            result=self.logic.infer(self.program,tokenize(torch.tensor(bits)))
            torch.testing.assert_close(result['action_activation'].sum(),torch.tensor(1.))
            torch.testing.assert_close(result['probs'].sum(),torch.tensor(1.))
            torch.testing.assert_close(result['support'],result['action_contribution'].sum(0))
    def test_hardened_program_has_exactly_one_true_path(self):
        program=self.program.harden()
        for bits in itertools.product([0.,1.],repeat=4):
            result=self.logic.infer(program,tokenize(torch.tensor(bits)))
            self.assertEqual(int((result['action_activation']>0).sum()),1)
            self.assertEqual(int((result['probs']>0).sum()),1)
    def test_disabling_only_applicable_rule_has_no_fallback(self):
        program=self.program.harden();state=tokenize(torch.zeros(4));result=self.logic.infer(program,state)
        active=int(result['action_activation'].argmax())+self.logic.action_slice.start
        with self.assertRaises(NoSupportedAction):self.logic.infer(program,state,disabled_rule=active)
    def test_raw_visual_or_soft_features_are_rejected(self):
        with self.assertRaises(TypeError):self.logic.infer(self.program,torch.rand(4))
    def test_gradients_pass_through_proof_to_learned_symbols_and_rules(self):
        observations=torch.rand(4,requires_grad=True)
        result=self.logic.infer(self.program,tokenize(observations))
        loss=-result['probs'][1].log();loss.backward()
        self.assertIsNotNone(observations.grad);self.assertGreater(float(observations.grad.abs().sum()),0)
        self.assertGreater(float(self.generator.head.connection_logits.grad.abs().sum()),0)
        self.assertGreater(float(self.generator.head.act.weight.grad.abs().sum()),0)
    def test_program_serialization_retains_interpreter_semantics(self):
        self.program.validate();copy=Program.from_record(self.program.record());copy.validate()
        self.assertEqual(copy.action_semantics,'branch_tree')
        report=describe(copy);self.assertEqual(len(report['nodes']),3);self.assertEqual(len(report['action_rules']),4)
        state=tokenize(torch.tensor([0.,1.,0.,1.]))
        torch.testing.assert_close(self.logic.infer(copy,state)['probs'],self.logic.infer(self.program,state)['probs'])
    def test_temperature_changes_only_generated_action_conclusions(self):
        other=SymbolicProgramGenerator(dict(self.cfg,route_temperature=4.),32)
        other.load_state_dict(self.generator.state_dict());p=other('automatically encoded task',[],[])
        torch.testing.assert_close(p.definitions,self.program.definitions)
        torch.testing.assert_close(p.premises,self.program.premises)
        torch.testing.assert_close(p.effects,self.program.effects)
        original=self.program.action_heads[self.logic.action_slice]
        expected=(original.clamp_min(1e-30).log()/4).softmax(-1)
        torch.testing.assert_close(p.action_heads[self.logic.action_slice],expected)
        p.validate();state=tokenize(torch.tensor([0.,1.,1.,0.]))
        result=self.logic.infer(p,state)
        torch.testing.assert_close(result['probs'],result['action_contribution'].sum(0))
    def test_invalid_temperature_rejected(self):
        for value in [0.,-1.,float('nan')]:
            with self.assertRaises(ValueError):SymbolicProgramGenerator(dict(self.cfg,route_temperature=value),32)


if __name__=='__main__':unittest.main()
