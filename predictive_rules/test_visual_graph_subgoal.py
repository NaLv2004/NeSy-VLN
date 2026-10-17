import unittest,torch
import test_graph_subgoal as original
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel,Config
from predictive_nesy.visual_graph_subgoal_training import objective

class VisualGraphTests(original.GraphTests):
    def setUp(self):
        super().setUp();self.model=VisualGraphSubgoalModel(Config(states=8,width=64,hidden=32,decoder_hidden=64))

    def test_future_action_tags_cannot_change_state_labels(self):
        p=self.program();first=self.model.training_targets(p,self.data['targets'],self.data['actions'])
        second=self.model.training_targets(p,self.data['targets'],torch.randint(0,17,self.data['actions'].shape))
        self.assertTrue(torch.equal(first,second))

    def test_identical_images_have_identical_state_labels(self):
        p=self.program();codes=self.data['targets'][:,:1].expand(-1,11,-1)
        states=self.model.training_targets(p,codes,self.data['actions'])
        self.assertTrue(torch.equal(states,states[:,:1].expand_as(states)))

    def test_visual_teacher_forcing_is_differentiable(self):
        loss,metrics=objective(self.model,self.data);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertIn('one_step_accuracy',metrics)
        for module in [self.model.observation,self.model.definitions,self.model.transitions,self.model.subgoal]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0.)

    def test_exported_rules_are_immutable_during_execution(self):
        from predictive_nesy.graph_subgoal_interpreter import export_program
        p=export_program(self.program())
        with self.assertRaises(ValueError):p.transitions[0,0,0]=0.

if __name__=='__main__':unittest.main()
