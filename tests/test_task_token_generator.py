import unittest
import torch
from nesy_vla.small_generator import SymbolicProgramGenerator
from nesy_vla.task_token_generator import TaskTokenProgramGenerator


class TaskTokenTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        self.cfg={'k':8,'d':32,'r':24,'a':5,'structured':True,'multi_effects':True}
        self.old=SymbolicProgramGenerator(self.cfg,32)
        self.model=TaskTokenProgramGenerator.initialize_from(self.old,self.cfg,['task beta','task alpha'],32)
    def test_preserves_learned_rules_and_only_adds_task_encoder(self):
        for name,value in self.old.head.state_dict().items():self.assertTrue(torch.equal(value,self.model.head.state_dict()[name]))
        self.assertEqual(self.model.task_vocabulary,['task alpha','task beta'])
    def test_context_is_target_sensitive_and_ignores_raw_images(self):
        a=self.model.context('task alpha',[],[]);b=self.model.context('task beta',[],[])
        self.assertFalse(torch.equal(a,b));self.assertTrue(torch.equal(a,self.model.context('task alpha',[object()],[])))
    def test_still_returns_learned_program(self):
        program=self.model('task alpha',[],[]);program.validate()
        self.assertEqual(tuple(program.definitions.shape),(8,32))
        self.assertEqual(tuple(program.action_heads.shape),(24,5))
        with self.assertRaises(ValueError):self.model.context('unseen task',[],[])
    def test_history_accepts_discrete_symbols_only(self):
        self.model.context('task alpha',[],[{'symbol_codes':[0,1]*4,'action':2}])
        with self.assertRaises(ValueError):self.model.context('task alpha',[],[{'symbol_codes':[.1]*8,'action':2}])


if __name__=='__main__':unittest.main()
