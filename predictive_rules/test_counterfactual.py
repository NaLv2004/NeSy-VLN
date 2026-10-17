import unittest
import torch
from predictive_nesy.model_memory import PredictiveMemoryModel,MemoryConfig
from predictive_nesy.counterfactual import objective,propagated_predictions


class CounterfactualTests(unittest.TestCase):
    def sample(self):
        b,t=2,4
        return dict(task=torch.tensor([0,1]),start=0,offset=t-1,
            history=torch.randint(0,64,(b,1,49)),previous_actions=torch.full((b,1),17),
            prefix_codes=torch.randint(0,64,(b,t,49)),prefix_actions=torch.randint(1,17,(b,t-1)),
            successors=torch.randint(0,64,(b,17,49)),collision=torch.rand(b,17)>.5,
            eligible_next=torch.rand(b,17)>.75,eligible_current=torch.zeros(b,dtype=torch.bool))

    def test_no_future_picture_can_correct_propagated_state(self):
        model=PredictiveMemoryModel(MemoryConfig(states=4,hidden=32))
        batch=self.sample();batch['prefix_codes'][:,0]=batch['history'][:,0]
        _,state,after=propagated_predictions(model,batch)
        batch['prefix_codes'][:,1:]=torch.randint(0,64,(2,3,49))
        batch['successors']=torch.randint(0,64,(2,17,49))
        _,changed_state,changed_after=propagated_predictions(model,batch)
        self.assertTrue(torch.equal(state,changed_state));self.assertTrue(torch.equal(after,changed_after))

    def test_observed_outcomes_train_generated_rules(self):
        model=PredictiveMemoryModel(MemoryConfig(states=4,hidden=32))
        batch=self.sample();batch['prefix_codes'][:,0]=batch['history'][:,0]
        loss,metrics=objective(model,batch);self.assertTrue(torch.isfinite(loss));loss.backward()
        for module in [model.transitions,model.risks,model.goals,model.memory_features]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)


if __name__=='__main__':unittest.main()
