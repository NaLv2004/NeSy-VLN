import itertools,unittest
import torch
from nesy_vla.branch_logic import TotalBranchLogic
from nesy_vla.counterfactual_learning import counterfactual_symbol_bc
from nesy_vla.small_generator import SymbolicProgramGenerator
from nesy_vla.symbols import tokenize


class CounterfactualLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(202611)
        self.generator=SymbolicProgramGenerator(dict(k=4,d=32,r=12,a=3,structured=True,
            multi_effects=True,action_semantics='branch_tree'),32)
        self.program=self.generator('learned task',[],[]);self.logic=TotalBranchLogic(4,3,12)

    def test_batched_execution_matches_every_scalar_proof(self):
        bits=torch.tensor(list(itertools.product([0.,1.],repeat=4)))
        batch=self.logic.probabilities_for_states(self.program,tokenize(bits))
        scalar=torch.stack([self.logic.infer(self.program,tokenize(row))['probs'] for row in bits])
        torch.testing.assert_close(batch,scalar)

    def test_grounding_gradient_equals_actual_one_bit_loss_differences(self):
        raw=torch.tensor([.2,.8,.1,.7],requires_grad=True);state=tokenize(raw);p=self.program.detach()
        loss=counterfactual_symbol_bc(self.logic,p,state,1);gradient=torch.autograd.grad(loss,raw)[0]
        actual=-self.logic.infer(p,state)['probs'][1].log();torch.testing.assert_close(loss,actual)
        expected=[]
        for index in range(4):
            bits=state.values.detach().clone();bits[index]=1-bits[index]
            alternate=-self.logic.infer(p,tokenize(bits))['probs'][1].log()
            expected.append((alternate-actual.detach())*(1-2*state.values[index].detach()))
        torch.testing.assert_close(gradient,torch.stack(expected))

    def test_rule_gradients_preserve_actual_likelihood_learning(self):
        bits=tokenize(torch.tensor([0.,1.,1.,0.]))
        parameters=[self.generator.head.act.weight,self.generator.head.connection_logits]
        cf=counterfactual_symbol_bc(self.logic,self.program,bits,2)
        ordinary=-self.logic.infer(self.program,bits)['probs'][2].log()
        actual=torch.autograd.grad(cf,parameters,retain_graph=True)
        expected=torch.autograd.grad(ordinary,parameters)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b)

    def test_continuous_input_is_rejected(self):
        with self.assertRaises(TypeError):self.logic.probabilities_for_states(self.program,torch.rand(3,4))
        with self.assertRaises(TypeError):counterfactual_symbol_bc(self.logic,self.program,torch.rand(4),0)

    def test_no_effect_produces_no_grounding_gradient(self):
        p=self.program.detach();p.action_heads[:]=torch.tensor([.2,.3,.5])
        raw=torch.rand(4,requires_grad=True)
        loss=counterfactual_symbol_bc(self.logic,p,tokenize(raw),1)
        gradient=torch.autograd.grad(loss,raw)[0]
        torch.testing.assert_close(gradient,torch.zeros_like(gradient),atol=1e-6,rtol=0)
