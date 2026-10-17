import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nesy_vla.generator import ProgramDecoder,sparse_literals,balanced_routes,pool_context
from nesy_vla.logic import SoftLogic
from nesy_vla.revision import successful_revision_winner


class StructuredTests(unittest.TestCase):
    def test_long_context_pool_matches_values_and_gradients(self):
        x=torch.randn(1,8193,8,requires_grad=True);y=x.detach().clone().requires_grad_()
        actual=pool_context(x);expected=torch.nn.functional.adaptive_avg_pool1d(y.transpose(1,2),256).transpose(1,2)
        self.assertTrue(torch.allclose(actual,expected,atol=2e-6))
        actual.square().sum().backward();expected.square().sum().backward()
        self.assertTrue(torch.allclose(x.grad,y.grad,atol=2e-6))

    def test_failed_stop_is_never_a_distillation_target(self):
        outcomes=[{'success':False,'actions':[4]*6},{'success':False,'actions':[0]}]
        self.assertIsNone(successful_revision_winner([-15.47,-12.04],outcomes))

    def test_success_beats_cheaper_failure(self):
        outcomes=[{'success':False,'actions':[4]*6},{'success':False,'actions':[0]},
                  {'success':True,'actions':[3,4,0]}]
        self.assertEqual(successful_revision_winner([-30,-12,-15],outcomes),2)

    def test_identical_candidates_are_not_new_evidence(self):
        self.assertIsNone(successful_revision_winner([25,25,25],[{'success':True,'actions':[0]}]*3))

    def test_premises_do_not_mix_a_literal_with_its_negation(self):
        x=torch.tensor([[[0.,10.,1.,9.,2.]]],requires_grad=True)
        p=sparse_literals(x,2)
        self.assertLessEqual(int((p>0).sum()),2)
        self.assertEqual(float(p[0,0,1]*p[0,0,3]),0.)
        (p*torch.arange(5)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_balanced_heads_cannot_all_route_to_one_action(self):
        torch.manual_seed(2);logits=torch.randn(32,17,requires_grad=True)
        logits=logits+torch.nn.functional.one_hot(torch.tensor(4),17)*8
        p=balanced_routes(logits)
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(32),atol=1e-5))
        self.assertLess(float((p.sum(0)-32/17).abs().max()),.1)
        self.assertGreater(len(set(p.argmax(-1).tolist())),8)

    def test_distinct_definitions_and_state_dependent_proofs(self):
        torch.manual_seed(6)
        h=ProgramDecoder(hidden=32,k=32,r=96,structured=True)
        p=h(torch.randn(1,20,32));p.validate()
        norm=torch.nn.functional.normalize(p.definitions,dim=-1)
        gram=norm@norm.T;off=gram[~torch.eye(32,dtype=torch.bool)]
        self.assertLess(float(off.abs().mean()),.15)
        logic=SoftLogic(32,17,96);actions=[]
        for _ in range(32):
            result=logic.infer(p,torch.rand(32),torch.ones(32))
            actions.append(int(result['probs'].argmax()))
            reconstructed=1-torch.prod(1-result['action_contribution'],0)
            self.assertTrue(torch.allclose(reconstructed,result['support'],atol=1e-6))
        self.assertGreater(len(set(actions)),1)

    def test_batched_symbolic_transitions_equal_each_action(self):
        torch.manual_seed(5)
        head=ProgramDecoder(hidden=32,k=32,r=96,structured=True)
        p=head(torch.randn(1,20,32));logic=SoftLogic(32,17,96)
        b=torch.rand(32);c=torch.rand(32);retention=torch.rand(17,32)
        predicted,confidence,_=logic.transition_all(p,b,c,retention)
        for a in range(17):
            expected,expected_c,_=logic.transition(p,b,c,a,retention[a])
            self.assertTrue(torch.allclose(predicted[a],expected,atol=1e-6))
            self.assertTrue(torch.allclose(confidence[a],expected_c,atol=1e-6))


if __name__=='__main__':unittest.main()
