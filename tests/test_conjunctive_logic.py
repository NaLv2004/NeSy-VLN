import itertools,json,unittest
import torch
from nesy_vla.small_generator import SymbolicProgramGenerator
from nesy_vla.conjunctive_logic import ConjunctiveRuleLogic
from nesy_vla.symbols import tokenize
from nesy_vla.program import Program,describe
from nesy_vla.logic import NoSupportedAction


class ConjunctiveLogicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(81)
        self.generator=SymbolicProgramGenerator(dict(k=4,d=32,r=12,a=3,structured=True,
            multi_effects=True,action_semantics='conjunctive_rules'),32)
        self.program=self.generator('automatically encoded task',[],[]);self.logic=ConjunctiveRuleLogic(4,3,12)

    def test_execution_matches_explicit_literal_conjunctions(self):
        p=self.program;bits=torch.tensor(list(itertools.product([0.,1.],repeat=4)))
        got=self.logic.probabilities_for_states(p,tokenize(bits))
        for row,probabilities in zip(bits,got):
            literal=torch.cat([torch.ones(1),row,1-row]);sel=self.logic.action_slice
            c=p.conjunctive_conditions[sel]
            weight=(p.premises[sel,0]@literal)*(c[:,:,0]+c[:,:,1]*row+c[:,:,2]*(1-row)).prod(-1)*p.weights[sel]
            expected=(weight[:,None]*p.action_heads[sel]).sum(0)/weight.sum()
            torch.testing.assert_close(probabilities,expected)
            proof=self.logic.infer(p,tokenize(row));torch.testing.assert_close(proof['probs'],expected)
            torch.testing.assert_close(proof['action_contribution'].sum(0),expected)

    def test_false_hard_conditions_have_no_fallback(self):
        p=self.program.harden();p.conjunctive_conditions[:]=torch.tensor([0.,0.,1.])
        with self.assertRaises(NoSupportedAction):self.logic.infer(p,tokenize(torch.ones(4)))

    def test_raw_features_are_rejected(self):
        with self.assertRaises(TypeError):self.logic.infer(self.program,torch.rand(4))
        with self.assertRaises(TypeError):self.logic.probabilities_for_states(self.program,torch.rand(2,4))

    def test_generated_conditions_and_grounding_are_trainable(self):
        raw=torch.rand(4,requires_grad=True)
        p=self.logic.infer(self.program,tokenize(raw))['probs'];(-p[1].log()).backward()
        self.assertGreater(float(raw.grad.abs().sum()),0)
        self.assertGreater(float(self.generator.head.effect.weight.grad.abs().sum()),0)

    def test_serialized_program_replays_identical_proof(self):
        self.program.validate();record=self.program.record();json.dumps(record,allow_nan=False)
        p=Program.from_record(record);p.validate();state=tokenize(torch.tensor([0.,1.,0.,1.]))
        torch.testing.assert_close(self.logic.infer(p,state)['probs'],self.logic.infer(self.program,state)['probs'])
        self.assertEqual(describe(p)['condition_order'],['unused','required_true','required_false'])
        json.dumps(describe(p),allow_nan=False)

    def test_legacy_program_record_remains_compatible(self):
        record=self.program.record();record.pop('conjunctive_conditions');record['action_semantics']='horn'
        old=Program.from_record(record);old.validate();self.assertIsNone(old.conjunctive_conditions)

    def test_minimum_agrees_with_explicit_godel_conjunction(self):
        p=self.program;p.conjunction_operator='minimum';p.validate()
        bits=torch.tensor(list(itertools.product([0.,1.],repeat=4)));sel=self.logic.action_slice
        for row in bits:
            literal=torch.cat([torch.ones(1),row,1-row]);c=p.conjunctive_conditions[sel]
            matched=c[:,:,0]+c[:,:,1]*row+c[:,:,2]*(1-row)
            weight=torch.minimum(p.premises[sel,0]@literal,matched.amin(-1))*p.weights[sel]
            expected=(weight[:,None]*p.action_heads[sel]).sum(0)/weight.sum()
            proof=self.logic.infer(p,tokenize(row))
            torch.testing.assert_close(proof['probs'],expected)
            torch.testing.assert_close(proof['action_contribution'].sum(0),expected)
        replay=Program.from_record(p.record());replay.validate()
        self.assertEqual(replay.conjunction_operator,'minimum')
        raw=torch.rand(4,requires_grad=True)
        loss=-self.logic.infer(p,tokenize(raw))['probs'][1].log();loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_minimum_rejects_false_conditions_and_continuous_states(self):
        p=self.program.harden();p.conjunction_operator='minimum';p.conjunctive_conditions[:]=torch.tensor([0.,0.,1.])
        with self.assertRaises(NoSupportedAction):self.logic.infer(p,tokenize(torch.ones(4)))
        with self.assertRaises(TypeError):self.logic.infer(p,torch.rand(4))

    def test_legacy_operator_default_and_unknown_operator_rejected(self):
        record=self.program.record();record.pop('conjunction_operator')
        self.assertEqual(Program.from_record(record).conjunction_operator,'product')
        self.program.conjunction_operator='unknown'
        with self.assertRaises(ValueError):self.program.validate()
