import inspect,tempfile,unittest,json
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.subgoal_model import SubgoalRuleModel,Config,straight_through
from predictive_nesy import subgoal_interpreter as logic
from predictive_nesy.subgoal_training import transition,all_transitions,future_prediction,objective
from predictive_nesy.subgoal_data import require_state_only,load_windows


class SubgoalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77);self.model=SubgoalRuleModel(Config(states=4,hidden=32))
        self.data=dict(task=torch.tensor([0,1]),start=torch.zeros(2,dtype=torch.long),history=torch.randint(0,64,(2,1,49)),
            previous_actions=torch.full((2,1),17),history_length=torch.ones(2,dtype=torch.long),targets=torch.randint(0,64,(2,11,49)),
            actions=torch.tensor([[1,2,0,0,0,0,0,0,0,0]]*2),action_valid=torch.tensor([[1,1,1,0,0,0,0,0,0,0]]*2).bool(),
            state_valid=torch.tensor([[1,1,1,1,0,0,0,0,0,0,0]]*2).bool(),supervised=torch.ones(2,10).bool(),goal_index=torch.tensor([3,3]))

    def test_no_reward_value_or_semantic_head(self):
        self.assertFalse(any(word in name for name in self.model.state_dict() for word in ['reward','value','risk','collision','actor']))
        program,_=future_prediction(self.model,self.data);exported=logic.export_program(program)
        self.assertEqual(set(exported.__dict__),set(logic.FIELDS)|{'start','interval'})
        self.assertEqual(exported.subgoal.tolist(),program['subgoal_logits'][0].argmax(-1).tolist())
        with self.assertRaises(ValueError):require_state_only({'scalar_rewards':torch.ones(1)})
        with tempfile.TemporaryDirectory() as d:
            exported.save(Path(d)/'p.npz');self.assertEqual(logic.Program.load(Path(d)/'p.npz').digest(),exported.digest())

    def test_exact_rule_execution_matches_training(self):
        program,_=future_prediction(self.model,self.data)
        states=straight_through(self.model.ground(program,self.data['targets'][:,0]))
        for action in range(17):
            predicted=transition(program,states,torch.full((2,),action)).argmax(-1)
            torch.testing.assert_close(all_transitions(program,states)[:,action],transition(program,states,torch.full((2,),action)))
            for i in range(2):
                after,_=logic.successors(logic.export_program(program,i),states[i].argmax(-1).numpy())
                np.testing.assert_array_equal(after[action],predicted[i].numpy())

    def test_future_observations_do_not_change_predicted_path(self):
        _,before=future_prediction(self.model,self.data)
        changed={**self.data,'targets':self.data['targets'].clone()};changed['targets'][:,1:]=torch.randint(0,64,(2,10,49))
        _,after=future_prediction(self.model,changed);torch.testing.assert_close(before,after,atol=0,rtol=0)

    def test_goal_and_dynamics_receive_gradients(self):
        loss,metrics=objective(self.model,self.data);loss.backward();self.assertTrue(torch.isfinite(loss))
        for module in [self.model.subgoal,self.model.transitions,self.model.observation]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)
        self.assertNotIn('scalar_return',metrics)

    @staticmethod
    def toy(reachable=True):
        f,a,k=2,3,4;t=np.zeros((f,a,k,k,k),np.float32)
        for j in range(f):
            for action in range(a):
                for before in range(k):
                    after=before
                    if j==0:
                        if before==0 and action==2:after=1
                        if before==1 and action==1:after=2
                        if reachable and before==2 and action==0:after=3
                    t[j,action,before,:,after]=1
        return logic.Program(parents=np.array([[0,0],[1,1]]),transitions=t,definitions=np.zeros((f,k,8)),subgoal=np.array([3,0]),start=0)

    def test_complete_symbolic_goal_proof(self):
        p=self.toy();c=logic.Controller();plan=c.revise(p,np.array([0,0]),horizon=10)
        self.assertTrue(plan['found']);self.assertEqual(plan['actions'],[2,1,0])
        self.assertEqual(plan['states'][-1],p.subgoal.tolist())
        for action in plan['actions']:
            chosen,proof=c.next_action();self.assertEqual(chosen,action)
        self.assertTrue(proof['subgoal_reached']);self.assertTrue(c.finished)

    def test_no_unproven_fallback_or_between_step_observation(self):
        c=logic.Controller();plan=c.revise(self.toy(False),np.array([0,0]))
        self.assertFalse(plan['found']);self.assertEqual(plan['actions'],[])
        with self.assertRaises(ValueError):c.next_action()
        self.assertEqual(list(inspect.signature(c.next_action).parameters),[])

    def test_boundary_neural_gate(self):
        from predictive_nesy.gate import NeuralBoundaryGate
        gate=NeuralBoundaryGate([self.model])
        try:
            with gate.boundary(0):future_prediction(self.model,self.data)
            with self.assertRaises(RuntimeError):future_prediction(self.model,self.data)
        finally:gate.close()

    def test_data_has_no_return_targets_and_respects_last_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);episode='P25_train_100_chair_test'
            rows=[dict(episode_id=episode,category='chair',label=1 if i<34 else 0) for i in range(35)]
            (folder/'training_rows.json').write_text(json.dumps(rows))
            (folder/'manifest.json').write_text(json.dumps(dict(data_provenance=dict(train_layouts=['layout_100']))))
            np.save(folder/'visual_tokens.npy',np.arange(35)[:,None].repeat(49,axis=1))
            windows,_=load_windows(folder,['layout_100'])
            self.assertEqual([w['start'] for w in windows],[0,10,20,30])
            self.assertEqual(windows[-1]['goal_index'],2)
            self.assertEqual(windows[-1]['action_valid'].sum(),2)
            for w in windows:require_state_only(w)
            np.testing.assert_array_equal(windows[-1]['targets'][2],np.full(49,32))


if __name__=='__main__':unittest.main()
