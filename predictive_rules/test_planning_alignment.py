import unittest
from itertools import combinations
import numpy as np
import torch
from predictive_nesy.planning_alignment import beam_candidates,sequence_scores,loss
from predictive_nesy import belief_interpreter as numpy_logic
from predictive_nesy.interpreter import export_program
from predictive_nesy.model_memory import PredictiveMemoryModel,MemoryConfig
from test_contract import toy_program


def tensors(p):
    return {**{name:torch.as_tensor(getattr(p,name)).float()[None] for name in ['transitions','goals','risks','values','weights']},
        'parents':torch.nn.functional.one_hot(torch.as_tensor(p.parents),len(p.parents)).float()[None]}


class PlanningAlignmentTests(unittest.TestCase):
    def test_full_toy_plan_and_return_match_standalone_interpreter(self):
        p=toy_program();program=tensors(p);initial=torch.nn.functional.one_hot(torch.tensor([[0,0]]),4).float()
        paths,scores=beam_candidates(program,initial,[(0,1)],horizon=10,beam_width=16)
        reference=numpy_logic.plan(p,initial[0].numpy(),horizon=10,beam_width=16)
        self.assertEqual(paths[0,0,:3].tolist(),[2,1,0])
        self.assertAlmostEqual(float(scores[0,0]),reference['score'],places=6)
        rescored=sequence_scores(program,initial,paths,[(0,1)])
        torch.testing.assert_close(scores,rescored,atol=1e-6,rtol=1e-6)

    def test_stochastic_program_matches_numpy_and_backpropagates(self):
        torch.manual_seed(13);m=PredictiveMemoryModel(MemoryConfig(states=4,hidden=32))
        codes=torch.randint(0,64,(2,1,49))
        p=m.generate(torch.tensor([0,1]),codes,torch.full((2,1),17),torch.ones(2,dtype=torch.long),start=0)
        initial=torch.nn.functional.one_hot(m.ground(p,codes[:,0]).argmax(-1),4).float()
        paths,scores=beam_candidates(p,initial,m.pairs,horizon=5,beam_width=8)
        for i in range(2):
            reference=numpy_logic.plan(export_program(p,i),initial[i].numpy(),horizon=5,beam_width=8)
            self.assertEqual(paths[i,0,:len(reference['actions'])].tolist(),reference['actions'])
            self.assertAlmostEqual(float(scores[i,0]),reference['score'],places=5)
        rescored=sequence_scores(p,initial,paths[:,:2],m.pairs)
        torch.testing.assert_close(scores[:,:2],rescored,atol=2e-6,rtol=2e-6)
        rescored.mean().backward()
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in m.transitions.parameters() if x.grad is not None),0)


if __name__=='__main__':unittest.main()
