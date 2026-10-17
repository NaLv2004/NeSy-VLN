import copy,unittest
import torch
from torch import nn
from nesy_vla.model import NavigationModel


class TeacherTargetTests(unittest.TestCase):
    def model(self,decay=None):
        model=NavigationModel.__new__(NavigationModel);nn.Module.__init__(model)
        model.cfg={} if decay is None else {'teacher_decay':decay}
        model.grounder=nn.Sequential(nn.Linear(4,4),nn.Linear(4,1))
        model.teacher=copy.deepcopy(model.grounder).requires_grad_(False)
        with torch.no_grad():
            for p in model.grounder.parameters():p.add_(1.)
        return model

    def test_default_preserves_ema(self):
        model=self.model();before=[p.clone() for p in model.teacher.parameters()]
        model.update_teacher()
        for old,target,current in zip(before,model.teacher.parameters(),model.grounder.parameters()):
            torch.testing.assert_close(target,old*.995+current*.005)

    def test_synchronous_targets_match_current_symbols(self):
        model=self.model(0.);inputs=torch.randn(20,4)
        model.update_teacher()
        for target,current in zip(model.teacher.parameters(),model.grounder.parameters()):
            torch.testing.assert_close(target,current,rtol=0,atol=0)
            self.assertFalse(target.requires_grad);self.assertIsNone(target.grad)
        torch.testing.assert_close((model.teacher(inputs)>=0).long(),(model.grounder(inputs)>=0).long())

    def test_explicit_override_and_shared_frozen_parameters(self):
        model=self.model(.9);model.teacher[0]=model.grounder[0]
        shared_before=model.grounder[0].weight.clone();model.update_teacher(0.)
        torch.testing.assert_close(shared_before,model.grounder[0].weight,rtol=0,atol=0)
        torch.testing.assert_close(model.teacher[1].weight,model.grounder[1].weight,rtol=0,atol=0)

    def test_invalid_decay_cannot_mutate_teacher(self):
        for decay in [-.1,1.,float('nan'),float('inf')]:
            model=self.model(decay);before=[p.clone() for p in model.teacher.parameters()]
            with self.assertRaises(ValueError):model.update_teacher()
            for old,current in zip(before,model.teacher.parameters()):torch.testing.assert_close(old,current,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
