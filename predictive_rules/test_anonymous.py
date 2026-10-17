import inspect
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.anonymous_model import AnonymousRuleModel,Config,straight_through,transition_all,reward_all
from predictive_nesy import anonymous_interpreter as logic
from predictive_nesy.anonymous_planning import beam_candidates,sequence_scores
from predictive_nesy.anonymous_training import outcome_predictions,outcome_objective,require_anonymous


class AnonymousTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41);self.m=AnonymousRuleModel(Config(states=4,hidden=32))
        self.codes=torch.randint(0,64,(2,2,49))

    def program(self):
        return self.m.generate(torch.tensor([0,1]),self.codes[:,:1],torch.full((2,1),17),torch.ones(2,dtype=torch.long),start=0)

    def test_only_anonymous_parameters_are_constructed_and_exported(self):
        p=self.program()
        self.assertEqual(set(p),set(logic.FIELDS)|{'start','context_raw'})
        self.assertFalse(any(word in key for key in self.m.state_dict() for word in ['risk','goal','collision']))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.npz';logic.export_program(p).save(path);q=logic.Program.load(path)
            self.assertEqual(q.digest(),logic.export_program(p).digest())
        with self.assertRaises(ValueError):require_anonymous({'collision':torch.zeros(2)})
        with self.assertRaises(ValueError):require_anonymous({'goal':torch.zeros(2)})

    def test_numpy_search_matches_tensor_search_and_scalar_returns(self):
        p=self.program();initial=straight_through(self.m.ground(p,self.codes[:,0]))
        paths,values=beam_candidates(p,initial,self.m.pairs,horizon=5,beam_width=8)
        for i in range(2):
            exported=logic.export_program(p,i);q=logic.plan(exported,initial[i].detach().numpy(),horizon=5,beam_width=8)
            self.assertEqual(paths[i,0,:len(q['actions'])].tolist(),q['actions'])
            self.assertAlmostEqual(float(values[i,0]),q['score'],places=5)
        rescored=sequence_scores(p,initial,paths[:,:2],self.m.pairs)
        torch.testing.assert_close(rescored,values[:,:2],atol=3e-6,rtol=3e-6)
        rescored.mean().backward()
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in self.m.transitions.parameters() if x.grad is not None),0.)
        self.assertFalse(any(x.grad is not None for x in self.m.rewards.parameters()))

    def test_future_images_only_supply_labels(self):
        b=dict(task=torch.tensor([0,1]),history=self.codes[:,:1],previous_actions=torch.full((2,1),17),
            prefix_codes=self.codes.clone(),prefix_actions=torch.tensor([[1],[2]]),successors=torch.randint(0,64,(2,17,49)),
            scalar_rewards=torch.zeros(2,17),start=0,offset=1)
        p,s,a=outcome_predictions(self.m,b)
        changed={**b,'prefix_codes':b['prefix_codes'].clone(),'successors':torch.randint(0,64,(2,17,49))}
        changed['prefix_codes'][:,1]=torch.randint(0,64,(2,49))
        _,s2,a2=outcome_predictions(self.m,changed)
        torch.testing.assert_close(s,s2,atol=0,rtol=0);torch.testing.assert_close(a,a2,atol=0,rtol=0)
        loss,_=outcome_objective(self.m,b);loss.backward();self.assertTrue(torch.isfinite(loss))

    def test_budget_exhaustion_does_not_reward_imaginary_future(self):
        p=self.program();initial=straight_through(self.m.ground(p,self.codes[:,0]))
        p['values']=torch.ones_like(p['values'])*.9
        paths,values=beam_candidates(p,initial,self.m.pairs,horizon=2,beam_width=32,terminal_horizon=True)
        for i in range(2):
            exported=logic.export_program(p,i)
            q=logic.plan(exported,initial[i].detach().numpy(),horizon=2,beam_width=32,terminal_horizon=True)
            self.assertEqual(paths[i,0,:len(q['actions'])].tolist(),q['actions'])
            self.assertAlmostEqual(float(values[i,0]),q['score'],places=5)
        rescored=sequence_scores(p,initial,paths[:,:2],self.m.pairs,terminal_horizon=True)
        torch.testing.assert_close(rescored,values[:,:2],atol=3e-6,rtol=3e-6)
        zero_value={**p,'values':torch.zeros_like(p['values'])}
        check=sequence_scores(zero_value,initial,paths[:,:2],self.m.pairs,terminal_horizon=True)
        torch.testing.assert_close(rescored,check,atol=0,rtol=0)
        p['start']=30;controller=logic.Controller();controller.step=30
        result=controller.revise(logic.export_program(p),initial[0].detach().numpy(),horizon=2)
        self.assertTrue(result['terminal_horizon'])

    def test_controller_has_no_observation_correction_channel(self):
        p=logic.export_program(self.program());controller=logic.Controller()
        state=np.zeros(8,np.int64);controller.revise(p,state,horizon=10)
        self.assertEqual(list(inspect.signature(controller.next_action).parameters),[])
        controller.next_action()
        with self.assertRaises(ValueError):controller.revise(p,state)

    def test_multistep_search_finds_delayed_return(self):
        f,a,k=2,3,4
        transitions=np.zeros((f,a,k,k,k),np.float32)
        for factor in range(f):
            for action in range(a):
                for before in range(k):
                    after=before
                    if factor==0 and action==2 and before==0:after=1
                    if factor==0 and action==1 and before==1:after=2
                    transitions[factor,action,before,:,after]=1
        rewards=np.full((1,a,k,k),-.02,np.float32);rewards[:,0]=-1;rewards[:,0,2,:]=1
        p=logic.Program(parents=np.array([[0,0],[1,1]]),transitions=transitions,rewards=rewards,
            values=np.zeros((1,k,k),np.float32),weights=np.ones(1,np.float32),definitions=np.zeros((f,k,8),np.float32),start=0)
        self.assertEqual(logic.plan(p,np.array([0,0]),horizon=3)['actions'],[2,1,0])
        self.assertNotEqual(logic.plan(p,np.array([0,0]),horizon=1)['actions'][0],2)

    def test_neural_gate_enforces_revision_only(self):
        from predictive_nesy.gate import NeuralBoundaryGate
        gate=NeuralBoundaryGate([self.m])
        try:
            with gate.boundary(0):self.program()
            with self.assertRaises(RuntimeError):self.program()
        finally:gate.close()

    def test_full_training_and_plan_loss_backward(self):
        from predictive_nesy.anonymous_training import objective
        from predictive_nesy.anonymous_planning import loss as plan_loss
        data=dict(task=torch.tensor([0,1]),start=torch.zeros(2,dtype=torch.long),history=self.codes[:,:1],
            previous_actions=torch.full((2,1),17),history_length=torch.ones(2,dtype=torch.long),
            targets=torch.randint(0,64,(2,11,49)),actions=torch.tensor([[1,2,0,0,0,0,0,0,0,0,0]]*2),
            valid=torch.tensor([[1,1,1,0,0,0,0,0,0,0,0]]*2).bool(),supervised=torch.ones(2,11).bool(),
            teacher_ids=torch.randint(0,4,(2,11,4)),scalar_rewards=torch.tensor([[-.02,-.02,1,1,1,1,1,1,1,1,1]]*2),
            returns=torch.ones(2,11))
        objective_loss,metrics=objective(self.m,data);objective_loss.backward()
        planning_loss,_=plan_loss(self.m,data,beam_width=4);planning_loss.backward()
        self.assertTrue(torch.isfinite(objective_loss+planning_loss))
        self.assertFalse(any(word in key for key in metrics for word in ['collision','goal','risk']))


if __name__=='__main__':unittest.main()
