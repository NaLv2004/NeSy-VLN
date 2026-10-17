import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nesy_vla.symbols import DiscreteState,tokenize
from nesy_vla.neural_logic import NeuralSymbolicLogic
from nesy_vla.small_generator import SymbolicProgramGenerator
from nesy_vla.generator import ProgramDecoder
from nesy_vla.logic import NoSupportedAction


class DiscreteTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(48)
        self.cfg=dict(k=16,d=64,r=48,a=17,structured=True)
        self.head=ProgramDecoder(hidden=32,**self.cfg)
        self.p=self.head(torch.randn(1,9,32))
        self.logic=NeuralSymbolicLogic(16,17,48)
        self.soft=torch.rand(16,requires_grad=True);self.state=tokenize(self.soft)

    def test_strict_discrete_boundary_and_gradient(self):
        self.assertEqual(set(self.state.codes.tolist()),{0,1})
        with self.assertRaises(TypeError):self.logic.infer(self.p,self.soft)
        with self.assertRaises(ValueError):self.logic.infer(self.p,DiscreteState(self.state.codes,self.soft))
        result=self.logic.infer(self.p,self.state)
        (-result['probs'][2].log()).backward()
        self.assertGreater(float(self.soft.grad.abs().sum()),0)
        self.assertGreater(float(self.head.connection_logits.grad.abs().sum()),0)
        self.assertGreater(float(self.logic.rule_gate[-1].weight.grad.abs().sum()),0)

    def test_combined_gate_batch_preserves_values_and_gradients(self):
        separate=torch.cat([self.logic.gates(self.p,self.state,s) for s in [*self.logic.layers,self.logic.action_slice]])
        combined=self.logic._validated_gates(self.p,self.state,slice(0,self.logic.action_slice.stop))
        self.assertTrue(torch.allclose(combined,separate,atol=1e-7))
        parameters=[self.soft,self.logic.rule_gate[0].weight,self.logic.rule_gate[-1].weight]
        a=torch.autograd.grad(combined.square().sum(),parameters,retain_graph=True)
        b=torch.autograd.grad(separate.square().sum(),parameters,retain_graph=True)
        for x,y in zip(a,b):self.assertTrue(torch.allclose(x,y,atol=2e-6,rtol=1e-5))

    def test_rule_proof_exact_and_no_fallback(self):
        result=self.logic.infer(self.p,self.state)
        from nesy_vla.logic import noisy_or
        self.assertTrue(torch.allclose(noisy_or(result['action_contribution'],0),result['support']))
        p=self.p.map(lambda t:t);p.weights=p.weights*0
        with self.assertRaises(NoSupportedAction):self.logic.infer(p,self.state)

    def test_neural_gates_cannot_fire_false_premises(self):
        p=self.p.map(lambda t:t);p.premises=p.premises*0
        p.premises[:,:,1]=1.
        zero=tokenize(torch.zeros(16));s=self.logic.fire_discrete(p,zero.values,torch.ones(16),self.logic.action_slice,zero)
        self.assertEqual(float(s.abs().sum()),0.)

    def test_discrete_truth_causal_effect(self):
        a=self.logic.infer(self.p,self.state)['probs']
        b=self.logic.infer(self.p,tokenize(1-self.soft))['probs']
        self.assertGreater(float((a-b).abs().sum()),.01)

    def test_transition_all_agrees_with_each_action(self):
        keep=torch.rand(17,16)
        predictions,_,_=self.logic.transition_all(self.p,self.soft,torch.ones(16),keep)
        for action in range(17):
            prediction,_,_=self.logic.transition(self.p,self.soft,torch.ones(16),action,keep[action])
            self.assertTrue(torch.allclose(prediction,predictions[action],atol=1e-6))

    def test_state_memory_is_discrete_and_can_change_rule_inputs(self):
        observed=tokenize(torch.ones(16));predicted=tokenize(torch.zeros(16))
        self.assertTrue(torch.equal(self.logic.assimilate(observed,predicted).codes,observed.codes))
        with torch.no_grad():self.logic.state_update[-1].bias.fill_(-1.)
        self.assertTrue(torch.equal(self.logic.assimilate(observed,predicted).codes,predicted.codes))
        with self.assertRaises(TypeError):self.logic.assimilate(observed,torch.rand(16))

    def test_generator_sees_only_symbols_and_text(self):
        g=SymbolicProgramGenerator(self.cfg)
        feedback=[{'symbol_codes':self.state.codes.tolist(),'action':4,'reward':100,'distance':.1}]
        p=g('Find an object',[object()],feedback)
        q=g('Find an object',[],[{**feedback[0],'reward':-500,'distance':50}])
        self.assertTrue(torch.equal(p.action_heads,q.action_heads))
        self.assertTrue(torch.equal(p.premises,q.premises))
        q=g('Find an object',[],[dict(feedback[0],symbol_codes=(1-self.state.codes).tolist())])
        self.assertGreater(float((p.premises-q.premises).abs().sum()),0.)
        with self.assertRaises(ValueError):g('Find an object',[],[{'symbol_codes':[.1]*16,'action':4}])
        revised=g('Find an object',[],feedback,p.detach());self.assertEqual(revised.version,1)


if __name__=='__main__':unittest.main()
