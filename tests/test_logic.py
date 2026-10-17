import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nesy_vla.generator import ProgramDecoder,program_distance
from nesy_vla.logic import SoftLogic,NoSupportedAction


class LogicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4);self.head=ProgramDecoder(hidden=32);self.logic=SoftLogic()
        self.context=torch.randn(1,20,32);self.p=self.head(self.context)
        self.b=torch.rand(16);self.c=torch.rand(16)

    def test_action_only_depends_on_program_and_belief(self):
        p=self.p;p.validate()
        result=self.logic.infer(p,self.b,self.c)
        reconstructed=1-torch.prod(1-result['action_contribution'],dim=0)
        self.assertTrue(torch.allclose(result['support'],reconstructed,atol=1e-6))
        self.assertAlmostEqual(float(result['probs'].sum()),1.,places=5)

    def test_zero_rules_never_fallback(self):
        p=self.p.map(lambda t:t);p.weights=p.weights*0
        with self.assertRaises(NoSupportedAction):self.logic.infer(p,self.b,self.c)

    def test_predicate_permutation_equivariance(self):
        perm=torch.randperm(16);p=self.p
        from nesy_vla.program import Program
        q=Program(p.definitions[perm],p.premises[:,:,torch.cat([torch.tensor([0]),1+perm,17+perm])],
            p.state_heads[:,perm],p.action_heads,p.effects[:,torch.cat([perm,16+perm])],p.action_conditions,p.weights)
        x=self.logic.infer(p,self.b,self.c)['probs'];y=self.logic.infer(q,self.b[perm],self.c[perm])['probs']
        self.assertTrue(torch.allclose(x,y,atol=1e-6))

    def test_gradient_through_program(self):
        result=self.logic.infer(self.p,self.b,self.c)
        (-result['probs'][3].log()).backward()
        self.assertGreater(float(self.head.context.weight.grad.abs().sum()),0)
        self.assertGreater(float(self.head.premise.weight.grad.abs().sum()),0)

    def test_revision_reads_old_program(self):
        revised=self.head(self.context,self.p.detach())
        self.assertEqual(revised.version,1)
        self.assertGreater(float(program_distance(revised,self.p).detach()),0)

    def test_unknown_is_not_strong_negative(self):
        literals=self.logic.literals(torch.full((16,),.5),torch.zeros(16))
        self.assertEqual(float(literals[1:].sum()),0.)

    def test_action_rule_requires_nonconstant(self):
        s=self.logic.action_slice
        self.assertEqual(float(self.p.premises[s,0,0].sum()),0.)

    def test_transition_is_bounded(self):
        b,c,conflict=self.logic.transition(self.p,self.b,self.c,4,torch.rand(16))
        self.assertTrue(bool(((b>=0)&(b<=1)&(c>=0)&(c<=1)).all()))
        self.assertGreaterEqual(float(conflict),0.)


if __name__=='__main__':unittest.main()
